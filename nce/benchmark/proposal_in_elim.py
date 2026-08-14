"""Proposal-sampling NN training inside FastGM elimination.

Adapts the proposal-sampling logic in `nce.benchmark.training.train_single_bucket`
(which expects a precomputed bucket .pt with exact_fw/exact_bw) to the
elimination-time setting where neither is available.

Used by `bucket.FastBucket.compute_message_nn` when `config['proposal_sampling']
is True`. Supports the same `proposal_mix` values: 'full' (pure WMB proposal),
'half' (50% uniform + 50% WMB), 'no_replacement', 'half_nr' (50% uniform +
50% NR).
"""
import math as _math
import time

import torch

from nce.training_logger import log_epoch_loss, log_val_loss
from nce.sampling import crn as _crn


def _is_loss_wrapper(trainer, base_loss_fn, loss_name):
    """Returns an importance-weighted loss fn that wraps the base loss."""
    def _compute_is_weights(proposal_lp):
        log_w = -proposal_lp
        log_w_shifted = log_w - log_w.max()
        w = torch.exp(log_w_shifted)
        w = w / w.sum() * len(w)
        return w.detach()

    def _is_loss(outputs, targets, bw_hat=None, _proposal_lp=None, **kwargs):
        if _proposal_lp is None:
            return base_loss_fn(outputs, targets, bw_hat, **kwargs)

        w = _compute_is_weights(_proposal_lp)

        if loss_name == 'neurobe_weighted_mse':
            epsilon = 1e-10
            ln_range = trainer.data_preprocessor.ln_max - trainer.data_preprocessor.ln_min
            sum_ln = trainer.data_preprocessor.sum_ln
            safe_sum_ln = sum_ln if abs(sum_ln) > epsilon else epsilon
            w_target = targets * ln_range / safe_sum_ln
            per_sample = w_target * (outputs - targets) ** 2
            return torch.mean(w * per_sample)

        # UKL (default)
        bw_normalizing_constant = trainer.data_preprocessor.bw_normalizing_constant
        max_val = trainer.data_preprocessor.global_max_targets
        if bw_hat is not None:
            bw_hat_d = bw_hat.detach()
            if bw_normalizing_constant is not None:
                bw_hat_d = bw_hat_d - bw_normalizing_constant
            outputs_adj = outputs + bw_hat_d
            targets_adj = targets + bw_hat_d
        else:
            outputs_adj = outputs
            targets_adj = targets

        if max_val is None:
            mv = torch.max(torch.max(targets_adj), torch.max(outputs_adj.detach()))
        else:
            mv = max(max_val, outputs_adj.detach().max().item())
            mv = torch.tensor(mv, device=outputs.device)
        mv = mv.detach()

        log_p = torch.clamp(targets_adj - mv, min=-100)
        log_q = torch.clamp(outputs_adj - mv, min=-100)
        p = torch.exp(log_p)
        q = torch.exp(log_q)
        per_sample = p * (log_p - log_q) - p + q
        return torch.sum(w * per_sample)

    return _is_loss


def _proposal_train_epoch(trainer, batches, loss_fn, config):
    """Custom epoch loop that handles proposal_log_probs in batches."""
    batch_losses = []
    grad_clip_norm = config.get('grad_clip_norm', None)
    for batch in batches:
        x_b = batch['x'].to(config['device'])
        y_b = batch['y'].to(config['device'])
        bw_b = batch.get('bw')
        if bw_b is not None:
            bw_b = bw_b.to(config['device'])
        plp = batch.get('proposal_log_probs')
        if plp is not None:
            plp = plp.to(config['device'])

        trainer.net.train()
        if isinstance(trainer.optimizer, list):
            for opt in trainer.optimizer:
                opt.zero_grad()
        else:
            trainer.optimizer.zero_grad()

        outputs = trainer.net(x_b)
        loss = loss_fn(outputs.reshape(-1), y_b, bw_b, _proposal_lp=plp)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), grad_clip_norm)
        if isinstance(trainer.optimizer, list):
            for opt in trainer.optimizer:
                opt.step()
        else:
            trainer.optimizer.step()
        batch_losses.append(loss.cpu().item())
        del x_b, y_b, bw_b, plp, loss
        torch.cuda.empty_cache()
    return torch.tensor(sum(batch_losses) / len(batch_losses))


def train_bucket_with_proposal_sampling(bucket, fastgm, trainer, config):
    """Train one bucket's NN using the proposal-sampling regime in `config`.

    Parameters
    ----------
    bucket : FastBucket
        Current bucket being processed (has approximate_upstream/downstream factors set).
    fastgm : FastGM
        Parent graphical model.
    trainer : Trainer
        Already constructed for this bucket. trainer.dataloader.bw_modifier may be set.
    config : dict
        Must include `proposal_sampling=True`. Optional `proposal_mix` and
        `proposal_temperature`. `num_samples`, `num_epochs`, `batch_size` are honored.

    Side effects
    ------------
    Trains `trainer.net` in place via a custom epoch loop using IS-weighted loss.
    Populates trainer.losses with (epoch, loss) tuples. Uses
    `trainer.data_preprocessor` for normalization / one-hot / max_targets state.
    """
    from nce.sampling.proposal_sampler import build_proposal_for_bucket
    import nce.sampling.no_replacement_sampler_v3  # attach sample_no_replacement_v3

    device = config['device']
    dtype = next(trainer.net.parameters()).dtype
    proposal_ecl = config.get('bw_ecl', 0)
    # Per-bucket resolution: config['num_samples'] may be the per-cluster formula
    # string "nbe,<eps>[,<n_min>]" (it is no longer frozen into an int by the first
    # NN bucket -- see FastBucket.resolve_num_samples).
    num_proposal_samples = bucket.get_num_samples()
    proposal_mix = config.get('proposal_mix', 'full')
    proposal_temperature = float(config.get('proposal_temperature', 1.0))

    msg_scope = trainer.sample_generator.message_scope
    domain_sizes = [fastgm.matching_var(v).states for v in msg_scope]

    print(f"[ProposalElim] Bucket {bucket.label}: building proposal tree "
          f"(ecl={proposal_ecl}, T={proposal_temperature}, mix={proposal_mix})")
    proposal_tree = build_proposal_for_bucket(
        bucket, fastgm, ecl=proposal_ecl, temperature=proposal_temperature
    )
    print(f"[ProposalElim] Proposal tree: {len(proposal_tree.levels)} levels, "
          f"vars={proposal_tree.variables}")

    # ---- Sample according to proposal_mix ----
    if proposal_mix == 'no_replacement':
        rng = _crn.no_replacement_generator(
            config, msg_scope, domain_sizes, device, _crn.DRAW_TRAIN)
        # Use recursive NR — non-recursive caps at frontier_size phase-2 samples
        # via clamp(max=1.0), which fails for peaked distributions.
        nr_samples, _, nr_eff_log_probs_log10 = \
            proposal_tree.sample_no_replacement_v3_recursive(
                num_proposal_samples, M=1, rng=rng, mode='save')
        assignments = torch.stack(
            [nr_samples[v] for v in msg_scope], dim=1
        ).to(device)
        proposal_log_probs_nat = nr_eff_log_probs_log10.to(
            device=device, dtype=torch.float32
        ) * _math.log(10)
        print(f"[ProposalElim] NR: got {assignments.shape[0]} samples")

    elif proposal_mix == 'half':
        n_half = num_proposal_samples // 2
        n_wmb = num_proposal_samples - n_half
        uniform_assignments = _crn.proposal_uniform(
            config, n_half, msg_scope, domain_sizes, device, _crn.DRAW_TRAIN)
        log_uniform_density = -sum(_math.log(d) for d in domain_sizes)
        uniform_log_probs = torch.full(
            (n_half,), log_uniform_density, device=device, dtype=torch.float32
        )

        wmb_samples_dict, wmb_log_probs_log10 = proposal_tree.sample(
            n_wmb, crn_key=_crn.proposal_tree_key(
                config, msg_scope, domain_sizes, _crn.DRAW_TRAIN))
        wmb_assignments = torch.stack(
            [wmb_samples_dict[v] for v in msg_scope], dim=1
        ).to(device)
        wmb_log_probs = wmb_log_probs_log10.to(
            device=device, dtype=torch.float32
        ) * _math.log(10)

        assignments = torch.cat([uniform_assignments, wmb_assignments], dim=0)
        proposal_log_probs_nat = torch.cat([uniform_log_probs, wmb_log_probs], dim=0)
        print(f"[ProposalElim] half: {n_half} uniform + {n_wmb} WMB samples")

    elif proposal_mix == 'half_nr':
        n_half = num_proposal_samples // 2
        n_nr = num_proposal_samples - n_half
        uniform_assignments = _crn.proposal_uniform(
            config, n_half, msg_scope, domain_sizes, device, _crn.DRAW_TRAIN)
        log_uniform_density = -sum(_math.log(d) for d in domain_sizes)
        uniform_log_probs = torch.full(
            (n_half,), log_uniform_density, device=device, dtype=torch.float32
        )

        rng = _crn.no_replacement_generator(
            config, msg_scope, domain_sizes, device, _crn.DRAW_TRAIN)
        nr_samples, _, nr_eff_log_probs_log10 = \
            proposal_tree.sample_no_replacement_v3_recursive(
                n_nr, M=1, rng=rng, mode='save')
        nr_assignments = torch.stack(
            [nr_samples[v] for v in msg_scope], dim=1
        ).to(device)
        nr_log_probs = nr_eff_log_probs_log10.to(
            device=device, dtype=torch.float32
        ) * _math.log(10)

        assignments = torch.cat([uniform_assignments, nr_assignments], dim=0)
        proposal_log_probs_nat = torch.cat([uniform_log_probs, nr_log_probs], dim=0)
        print(f"[ProposalElim] half_nr: {n_half} uniform + {nr_assignments.shape[0]} NR")

    elif proposal_mix == 'uniform':
        # Pure uniform sampling, no proposal/NR cost. Used with
        # correction_proposal_mix='nr' or 'half_nr' to keep training cheap
        # while doing correction on harder-to-reach samples.
        assignments = _crn.proposal_uniform(
            config, num_proposal_samples, msg_scope, domain_sizes, device,
            _crn.DRAW_TRAIN)
        log_uniform_density = -sum(_math.log(d) for d in domain_sizes)
        proposal_log_probs_nat = torch.full(
            (num_proposal_samples,), log_uniform_density,
            device=device, dtype=torch.float32
        )
        print(f"[ProposalElim] uniform: {num_proposal_samples} samples")

    else:
        # 'full' — pure proposal sampling
        samples_dict, proposal_log_probs = proposal_tree.sample(
            num_proposal_samples, crn_key=_crn.proposal_tree_key(
                config, msg_scope, domain_sizes, _crn.DRAW_TRAIN))
        assignments = torch.stack(
            [samples_dict[v] for v in msg_scope], dim=1
        ).to(device)
        proposal_log_probs_nat = proposal_log_probs.to(
            device=device, dtype=torch.float32
        ) * _math.log(10)
        print(f"[ProposalElim] full: {num_proposal_samples} samples")

    # ---- Build training data ----
    y_log10 = trainer.sample_generator.compute_message_values(assignments)
    bw_log10 = None
    if hasattr(trainer.dataloader, 'bw_modifier') and trainer.dataloader.bw_modifier is not None:
        bw_log10 = trainer.sample_generator.compute_backward_values(
            assignments, backward_factors=[trainer.dataloader.bw_modifier]
        )
    elif hasattr(trainer.dataloader, 'bw_factors') and trainer.dataloader.bw_factors is not None:
        bw_log10 = trainer.sample_generator.compute_backward_values(
            assignments, backward_factors=trainer.dataloader.bw_factors
        )

    trainer.data_preprocessor._initialize_normalizing_constant(y_log10, bw_log10)
    y_all, bw_all = trainer.data_preprocessor.normalize(y_log10, bw_log10)
    x_all = trainer.data_preprocessor.one_hot_encode(bucket, assignments)
    proposal_log_probs_all = proposal_log_probs_nat.to(dtype=dtype)

    # Cast to target precision
    if dtype == torch.float64:
        x_all = x_all.to(dtype=dtype)
        y_all = y_all.to(dtype=dtype)
        if bw_all is not None:
            bw_all = bw_all.to(dtype=dtype)

    # Replace -inf for neurobe_weighted_mse
    if config.get('loss_fn', '') == 'neurobe_weighted_mse':
        n_inf = torch.isinf(y_all).sum().item()
        if n_inf > 0:
            y_all = torch.clamp(y_all, min=-10.0)
            print(f"[ProposalElim] clamped {n_inf} -inf targets to -10")

    # global_max_targets for UKL stability
    if config.get('loss_fn', '') == 'unnormalized_kl':
        with torch.no_grad():
            if bw_all is not None and trainer.data_preprocessor.bw_normalizing_constant is not None:
                tgt_for_max = y_all + bw_all - trainer.data_preprocessor.bw_normalizing_constant
            else:
                tgt_for_max = y_all
            trainer.data_preprocessor.global_max_targets = tgt_for_max.max().item()

    # NeuroBE-style patience early stopping. By default val_set = train_set
    # (training loss as patience signal). If `holdout_val_frac > 0`, a random
    # fraction of samples (with IS weights attached) is set aside as held-out
    # val, and patience is computed on val loss instead — this attacks
    # per-bucket overfitting (train loss → 1e-7 while Z error grows).
    use_neurobe_early_stop = config.get('neurobe_early_stopping', False)
    neurobe_stop_iter = int(config.get('neurobe_stop_iter', 2))
    holdout_val_frac = float(config.get('holdout_val_frac', 0.0))
    use_holdout_val = holdout_val_frac > 0 and len(x_all) > 100
    restore_best_val = bool(config.get('restore_best_val', True))
    if use_holdout_val:
        n_val = int(holdout_val_frac * len(x_all))
        # Keyed on the separator + the run seed, not on the bucket label
        # (an execution artefact) -- and the run seed was missing
        # entirely, so two seeds got the SAME train/val split.
        gen = torch.Generator(device=x_all.device).manual_seed(
            _crn.derive_seed('holdout-val-split',
                             run_seed=int(config.get('seed', 42)),
                             sep=_crn.stream_payload(
                                 int(config.get('seed', 42)), msg_scope,
                                 [int(d) for d in domain_sizes],
                                 'holdout', 0),
                             n=int(len(x_all))))
        perm = torch.randperm(len(x_all), device=x_all.device, generator=gen)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        x_val = x_all[val_idx]; y_val = y_all[val_idx]
        bw_val = bw_all[val_idx] if bw_all is not None else None
        plp_val = proposal_log_probs_all[val_idx]
        x_train = x_all[train_idx]; y_train = y_all[train_idx]
        bw_train = bw_all[train_idx] if bw_all is not None else None
        plp_train = proposal_log_probs_all[train_idx]
        print(f"[ProposalElim] held-out val: {n_val} val / {len(x_train)} train")
    else:
        x_train, y_train = x_all, y_all
        bw_train = bw_all
        plp_train = proposal_log_probs_all
        x_val = y_val = bw_val = plp_val = None

    # Build training batches
    batch_size = int(config.get('batch_size', len(x_train)))
    num_batches = (len(x_train) + batch_size - 1) // batch_size

    stratify = config.get('stratify_samples', False)
    if stratify and num_batches > 1 and len(x_train) > num_batches:
        # Mirror data_loader.py:108-147: ensure each batch gets one of the
        # top-`num_batches` samples (by y value), rest randomly filled with
        # non-top samples. Spreads peak-value points across mini-batches.
        top_indices = torch.topk(y_train, num_batches).indices
        all_indices = torch.arange(len(y_train), device=y_train.device)
        mask = torch.ones(len(y_train), dtype=torch.bool, device=y_train.device)
        mask[top_indices] = False
        other = all_indices[mask]
        other = other[torch.randperm(len(other), device=other.device)]
        batches = []
        other_pos = 0
        per_batch_other = batch_size - 1
        for i in range(num_batches):
            top_i = top_indices[i].unsqueeze(0)
            end_other = min(other_pos + per_batch_other, len(other))
            batch_other = other[other_pos:end_other]
            other_pos = end_other
            idx = torch.cat([top_i, batch_other])
            batches.append({
                'x': x_train[idx],
                'y': y_train[idx],
                'bw': bw_train[idx] if bw_train is not None else None,
                'proposal_log_probs': plp_train[idx],
            })
    else:
        batches = []
        for i in range(0, len(x_train), batch_size):
            end = min(i + batch_size, len(x_train))
            batches.append({
                'x': x_train[i:end],
                'y': y_train[i:end],
                'bw': bw_train[i:end] if bw_train is not None else None,
                'proposal_log_probs': plp_train[i:end],
            })

    # ---- Wrap loss + epoch ----
    loss_name = config.get('loss_fn', '')
    is_loss_fn = _is_loss_wrapper(trainer, trainer.loss_fn, loss_name)
    trainer.loss_fn = is_loss_fn

    # ---- Custom epoch loop ----
    num_epochs = int(config.get('num_epochs', 100))
    losses = []
    val_losses = []
    prev_best_val = float('inf')
    patience = 0
    best_state = None
    t0 = time.time()
    # UKL-style absolute-loss early stop (mirrors train.py:632). Enabled
    # whenever skip_early_stopping is False. Threshold configurable via
    # `loss_threshold` (default 0.0001 — same as train.py's standard).
    #
    # Optionally, an interpolating threshold: if both
    # `loss_threshold_log10_start` and `loss_threshold_log10_end` are set,
    # the threshold for epoch t is 10^(linear-interp in log10 from start
    # to end as t goes 0 → num_epochs). Lets the criterion be strict
    # early (rarely trips) and lenient late (catches plateaus).
    use_loss_threshold_stop = not config.get('skip_early_stopping', False)
    loss_threshold_fixed = float(config.get('loss_threshold', 0.0001))
    log10_thr_start = config.get('loss_threshold_log10_start')
    log10_thr_end = config.get('loss_threshold_log10_end')
    use_interp_threshold = (
        log10_thr_start is not None and log10_thr_end is not None
        and num_epochs > 0
    )

    gm_logger = getattr(bucket.gm, '_training_logger', None)

    for epoch in range(1, num_epochs + 1):
        loss = _proposal_train_epoch(trainer, batches, is_loss_fn, config)
        loss_val = float(loss.item() if hasattr(loss, 'item') else loss)
        losses.append((epoch, loss_val))
        if gm_logger is not None:
            log_epoch_loss(gm_logger, bucket.label, epoch, loss_val)
            # val_set = train_set in NR setting, so val loss = train loss.
            log_val_loss(gm_logger, bucket.label, epoch, loss_val)
        if trainer.use_scheduler and getattr(trainer, 'scheduler', None) is not None:
            if isinstance(trainer.scheduler, list):
                for s in trainer.scheduler:
                    s.step()
            else:
                trainer.scheduler.step()

        # UKL absolute-loss threshold early stop.
        if use_loss_threshold_stop:
            if use_interp_threshold:
                frac = epoch / num_epochs
                log10_thr = (float(log10_thr_start)
                             + (float(log10_thr_end) - float(log10_thr_start)) * frac)
                threshold = 10.0 ** log10_thr
            else:
                threshold = loss_threshold_fixed
            if loss_val < threshold:
                print(f"[ProposalElim] Loss {loss_val:.4e} below threshold "
                      f"{threshold:.4e} at epoch {epoch}; stopping.",
                      flush=True)
                break

        # Plateau-window early stop: if relative range over the last K epochs
        # is below a small threshold, stop. Optional config:
        #   plateau_window: K (int, default disabled)
        #   plateau_rel_range: float (default 0.005 = 0.5%)
        #   plateau_min_epochs: minimum epoch before plateau can trigger
        #     (default = plateau_window). Prevents firing on the initial
        #     slow-changing phase before training has even converged.
        plateau_window = config.get('plateau_window')
        if plateau_window is not None and len(losses) >= int(plateau_window):
            K = int(plateau_window)
            min_epoch = int(config.get('plateau_min_epochs', K))
            if epoch >= min_epoch:
                rel_thresh = float(config.get('plateau_rel_range', 0.005))
                recent = [v for _, v in losses[-K:]]
                mn, mx = min(recent), max(recent)
                mean = sum(recent) / len(recent)
                rel_range = (mx - mn) / max(abs(mean), 1e-12)
                if rel_range < rel_thresh:
                    print(f"[ProposalElim] Plateau stop at epoch {epoch}: "
                          f"window-{K} rel_range={rel_range:.4e} < {rel_thresh}",
                          flush=True)
                    break

        # NeuroBE patience early stopping. If `holdout_val_frac > 0`, the
        # patience signal is val loss (computed each epoch on the held-out
        # subset). Otherwise val_set = train_set and the signal is train loss.
        if use_neurobe_early_stop:
            if use_holdout_val:
                trainer.net.eval()
                with torch.no_grad():
                    x_b = x_val.to(config['device'])
                    y_b = y_val.to(config['device'])
                    bw_b = bw_val.to(config['device']) if bw_val is not None else None
                    plp_b = plp_val.to(config['device'])
                    out_v = trainer.net(x_b)
                    val_loss = float(is_loss_fn(
                        out_v.reshape(-1), y_b, bw_b, _proposal_lp=plp_b
                    ).item())
                val_losses.append((epoch, val_loss))
                if gm_logger is not None:
                    log_val_loss(gm_logger, bucket.label, epoch, val_loss)
                signal = val_loss
            else:
                signal = loss_val
            if signal < prev_best_val:
                prev_best_val = signal
                patience = 0
                if use_holdout_val and restore_best_val:
                    best_state = {
                        k: v.detach().clone()
                        for k, v in trainer.net.state_dict().items()
                    }
            else:
                patience += 1
            if patience > neurobe_stop_iter:
                tag = "val" if use_holdout_val else "train"
                print(f"[ProposalElim] NeuroBE patience stop at epoch {epoch}: "
                      f"patience {patience} > stop_iter {neurobe_stop_iter} "
                      f"({tag}_loss), best={prev_best_val:.4e}, "
                      f"current={signal:.4e}",
                      flush=True)
                if best_state is not None:
                    trainer.net.load_state_dict(best_state)
                    print(f"[ProposalElim] restored best-val weights",
                          flush=True)
                break

    # Phase 2: full-batch refinement. After Phase 1 ES triggers, switch to
    # a single full-batch and run a short patience loop. Bias terms shift
    # every prediction by a constant, so they directly translate into
    # log_Z error; mini-batch noise can prevent the bias from settling.
    #
    # UKL loss is `torch.sum(w * per_sample)` with `sum(w)=N`, so it scales
    # linearly with N. To keep the effective step size comparable to
    # Phase 1, we scale LR down by (mini_batch_size / full_batch_size).
    #
    # Config: final_full_batch_patience (int, e.g. 5). 0 = disabled.
    fb_patience = int(config.get('final_full_batch_patience', 0))
    if fb_patience > 0 and len(x_train) > 0:
        fb_batch = {
            'x': x_train,
            'y': y_train,
            'bw': bw_train,
            'proposal_log_probs': plp_train,
        }
        fb_max_epochs = int(config.get('final_full_batch_max_epochs', 1000))
        # Scale LR down so effective step matches Phase 1.
        scale = float(batch_size) / float(len(x_train))
        optimizers = (trainer.optimizer if isinstance(trainer.optimizer, list)
                      else [trainer.optimizer])
        saved_lrs = []
        for opt in optimizers:
            saved_lrs.append([g['lr'] for g in opt.param_groups])
            for g in opt.param_groups:
                g['lr'] = g['lr'] * scale
        print(f"[ProposalElim] Bucket {bucket.label}: phase 2 lr scaled by "
              f"{scale:.4e} (bs={batch_size}, full={len(x_train)})",
              flush=True)
        fb_best = float('inf')
        fb_pat = 0
        fb_t0 = time.time()
        fb_done = 0
        for fb_ep in range(1, fb_max_epochs + 1):
            loss = _proposal_train_epoch(trainer, [fb_batch], is_loss_fn, config)
            lv = float(loss.item() if hasattr(loss, 'item') else loss)
            losses.append((epochs_done := (losses[-1][0] if losses else 0) + 1, lv))
            if gm_logger is not None:
                log_epoch_loss(gm_logger, bucket.label, epochs_done, lv)
                log_val_loss(gm_logger, bucket.label, epochs_done, lv)
            fb_done = fb_ep
            if lv < fb_best:
                fb_best = lv
                fb_pat = 0
            else:
                fb_pat += 1
            if fb_pat > fb_patience:
                break
        # Restore original LRs
        for opt, lrs in zip(optimizers, saved_lrs):
            for g, lr in zip(opt.param_groups, lrs):
                g['lr'] = lr
        print(f"[ProposalElim] Bucket {bucket.label}: full-batch refine "
              f"{fb_done} epochs in {time.time()-fb_t0:.1f}s, "
              f"best_fb_loss={fb_best:.4e}, final={lv:.4e}",
              flush=True)

    # Optional: compute per-bucket "local error" correction signal.
    # log_A = logsumexp_x [ y_ln + bw_ln ]   (natural log of sum of f_exact*bw_approx)
    # log_B = logsumexp_x [ net_ln + bw_ln ] (natural log of sum of f_approx*bw_approx)
    # error = log_A - log_B  (signed, natural log; converted to log10 in output)
    # Stored on bucket.gm._local_errors for the runner to aggregate.
    # Also computes IS-weighted versions (using -proposal_log_prob as log w).
    #
    # If `correction_proposal_mix` is set in config (e.g. 'no_replacement',
    # 'half_nr', 'full'), regenerate samples with that mix AFTER training and
    # compute the local-error on those new samples instead. Lets us train
    # cheaply on uniform but correct using NR-targeted samples.
    if config.get('compute_local_error', True):
        try:
            corr_mix = config.get('correction_proposal_mix', None)
            corr_n = int(config.get('correction_num_samples', num_proposal_samples))
            _corr_raw_log10 = False
            if corr_mix is None or corr_mix == proposal_mix:
                # Use training samples for correction (normalized via preprocessor)
                x_corr = x_train
                y_corr = y_train
                bw_corr = bw_train
                plp_corr = plp_train
                corr_mix_tag = proposal_mix
                _corr_raw_log10 = False
            else:
                # Regenerate samples using correction_proposal_mix
                if corr_mix == 'no_replacement':
                    rng = _crn.no_replacement_generator(
                        config, msg_scope, domain_sizes, device,
                        _crn.DRAW_CORRECTION)
                    nr_samples, _, nr_eff_log_probs_log10 = \
                        proposal_tree.sample_no_replacement_v3_recursive(
                            corr_n, M=1, rng=rng, mode='save')
                    corr_assignments = torch.stack(
                        [nr_samples[v] for v in msg_scope], dim=1
                    ).to(device)
                    corr_plp_nat = nr_eff_log_probs_log10.to(
                        device=device, dtype=torch.float32
                    ) * _math.log(10)
                elif corr_mix == 'half_nr':
                    n_half = corr_n // 2
                    n_nr = corr_n - n_half
                    u_assignments = _crn.proposal_uniform(
                        config, n_half, msg_scope, domain_sizes, device,
                        _crn.DRAW_CORRECTION)
                    log_uniform_density = -sum(_math.log(d) for d in domain_sizes)
                    u_log_probs = torch.full(
                        (n_half,), log_uniform_density, device=device, dtype=torch.float32
                    )
                    rng = _crn.no_replacement_generator(
                        config, msg_scope, domain_sizes, device,
                        _crn.DRAW_CORRECTION)
                    nr_samples, _, nr_eff_log_probs_log10 = \
                        proposal_tree.sample_no_replacement_v3_recursive(
                            n_nr, M=1, rng=rng, mode='save')
                    nr_assignments = torch.stack(
                        [nr_samples[v] for v in msg_scope], dim=1
                    ).to(device)
                    nr_lp = nr_eff_log_probs_log10.to(
                        device=device, dtype=torch.float32
                    ) * _math.log(10)
                    corr_assignments = torch.cat([u_assignments, nr_assignments], dim=0)
                    corr_plp_nat = torch.cat([u_log_probs, nr_lp], dim=0)
                else:  # 'full'
                    samples_dict, corr_plp_log10 = proposal_tree.sample(
                        corr_n, crn_key=_crn.proposal_tree_key(
                            config, msg_scope, domain_sizes,
                            _crn.DRAW_CORRECTION))
                    corr_assignments = torch.stack(
                        [samples_dict[v] for v in msg_scope], dim=1
                    ).to(device)
                    corr_plp_nat = corr_plp_log10.to(
                        device=device, dtype=torch.float32
                    ) * _math.log(10)
                # Compute raw y, bw at correction samples (no preprocessor normalize).
                # We do un-normalization manually to support minmax_01 mode.
                y_corr_log10 = trainer.sample_generator.compute_message_values(corr_assignments)
                bw_corr_log10 = None
                if hasattr(trainer.dataloader, 'bw_modifier') and trainer.dataloader.bw_modifier is not None:
                    bw_corr_log10 = trainer.sample_generator.compute_backward_values(
                        corr_assignments, backward_factors=[trainer.dataloader.bw_modifier])
                elif hasattr(trainer.dataloader, 'bw_factors') and trainer.dataloader.bw_factors is not None:
                    bw_corr_log10 = trainer.sample_generator.compute_backward_values(
                        corr_assignments, backward_factors=trainer.dataloader.bw_factors)
                else:
                    # bw factors not attached (e.g. use_bw_approx=False). Build fresh.
                    from nce.utils.backward_message import get_backward_message
                    backward_iB = config.get('backward_iB', config.get('iB', 100))
                    backward_ecl = config.get('bw_ecl', config.get('ecl', 2**20))
                    use_precomputed = bucket.gm.populate_bw_factors and bucket.approximate_downstream_factors is not None
                    bf_arg = bucket.approximate_downstream_factors if use_precomputed else None
                    bw_factors_local, _ = get_backward_message(
                        bucket.gm, bucket.label,
                        backward_factors=bf_arg,
                        iB=backward_iB, backward_ecl=backward_ecl,
                        approximation_method='wmb', return_factor_list=True
                    )
                    bw_corr_log10 = trainer.sample_generator.compute_backward_values(
                        corr_assignments, backward_factors=bw_factors_local)

                x_corr = trainer.data_preprocessor.one_hot_encode(bucket, corr_assignments)
                if dtype == torch.float64:
                    x_corr = x_corr.to(dtype=dtype)
                # Skip preprocessor.normalize — use raw log10 values directly
                y_corr_log10 = y_corr_log10
                bw_corr_log10 = bw_corr_log10
                plp_corr = corr_plp_nat.to(dtype=dtype)
                corr_mix_tag = corr_mix
                print(f"[ProposalElim] Bucket {bucket.label}: correction "
                      f"samples regenerated with mix='{corr_mix}', n={len(x_corr)}",
                      flush=True)
                # Override y_corr/bw_corr/plp_corr to use raw log10 values
                y_corr = y_corr_log10
                bw_corr = bw_corr_log10
                plp_corr = corr_plp_nat
                _corr_raw_log10 = True

            if bw_corr is None:
                print(f"[ProposalElim] Bucket {bucket.label}: no bw, skipping local-err",
                      flush=True)
            else:
                with torch.no_grad():
                    trainer.net.eval()
                    x_dev = x_corr.to(config['device'])
                    net_normalized = trainer.net(x_dev).reshape(-1).float()
                    mode = getattr(trainer.data_preprocessor, 'normalization_mode', 'standard')
                    ln10 = _math.log(10.0)
                    if _corr_raw_log10:
                        # We have raw log10 values for y_corr and bw_corr
                        y_ln = y_corr.float() * ln10
                        bw_ln = bw_corr.float() * ln10
                        # Un-normalize net output
                        if mode == 'minmax_01':
                            ln_min = float(trainer.data_preprocessor.ln_min)
                            ln_range = float(trainer.data_preprocessor.ln_range)
                            net_ln = ln_min + net_normalized * ln_range
                        else:
                            norm_const_v = float(trainer.data_preprocessor.normalizing_constant)
                            net_ln = net_normalized + norm_const_v
                    else:
                        # Training-samples path: arrays were normalized via preprocessor
                        # Only standard mode supported here (minmax_01 drops bw)
                        norm_const_v = float(trainer.data_preprocessor.normalizing_constant)
                        y_ln = y_corr.float() + norm_const_v
                        net_ln = net_normalized + norm_const_v
                        bw_ln = bw_corr.float()
                    valid = torch.isfinite(y_ln) & torch.isfinite(bw_ln) & torch.isfinite(net_ln)
                    y_ln = y_ln[valid]
                    net_ln = net_ln[valid]
                    bw_ln = bw_ln[valid]
                    log_w = (-plp_corr.float())[valid]
                    log_A_raw = torch.logsumexp(y_ln + bw_ln, dim=0)
                    log_B_raw = torch.logsumexp(net_ln + bw_ln, dim=0)
                    log_A_is = torch.logsumexp(log_w + y_ln + bw_ln, dim=0)
                    log_B_is = torch.logsumexp(log_w + net_ln + bw_ln, dim=0)
                import math as _m
                ln10 = _m.log(10.0)
                entry = {
                    'bucket': int(bucket.label),
                    'corr_mix': corr_mix_tag,
                    'n_used': int(valid.sum().item()),
                    'log_A_raw_log10': float(log_A_raw.item()) / ln10,
                    'log_B_raw_log10': float(log_B_raw.item()) / ln10,
                    'log_A_IS_log10':  float(log_A_is.item())  / ln10,
                    'log_B_IS_log10':  float(log_B_is.item())  / ln10,
                }
                entry['err_raw_log10'] = entry['log_A_raw_log10'] - entry['log_B_raw_log10']
                entry['err_IS_log10']  = entry['log_A_IS_log10']  - entry['log_B_IS_log10']
                if not hasattr(bucket.gm, '_local_errors'):
                    bucket.gm._local_errors = []
                bucket.gm._local_errors.append(entry)
                print(f"[ProposalElim] Bucket {bucket.label}: local-err "
                      f"({corr_mix_tag}) raw={entry['err_raw_log10']:+.5f}, "
                      f"IS={entry['err_IS_log10']:+.5f} (log10)",
                      flush=True)
        except Exception as e:
            import traceback as _tb
            print(f"[ProposalElim] local-error compute failed: {e}", flush=True)
            print(_tb.format_exc(), flush=True)

    trainer.losses = losses
    trainer.val_losses = val_losses
    epochs_done = losses[-1][0] if losses else 0
    print(f"[ProposalElim] Bucket {bucket.label}: trained {epochs_done} epochs in "
          f"{time.time()-t0:.1f}s, final_loss={losses[-1][1]:.4e}")
