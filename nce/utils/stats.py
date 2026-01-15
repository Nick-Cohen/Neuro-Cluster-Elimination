#%%
import torch, math, copy, sys, os, logging
from contextlib import contextmanager

@contextmanager
def mute_everything():
    # Save state
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    stdout_fd, stderr_fd = os.dup(1), os.dup(2)  # OS-level
    prev_logging_disable = logging.root.manager.disable
    prev_tqdm_disable = os.environ.get("TQDM_DISABLE")

    devnull = open(os.devnull, "w")
    try:
        # Mute logging and tqdm
        logging.disable(logging.CRITICAL)
        os.environ["TQDM_DISABLE"] = "1"

        # Flush and redirect OS-level fds
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)

        # Point Python-level streams at devnull too
        sys.stdout = devnull
        sys.stderr = devnull

        yield
    finally:
        # Restore everything
        try:
            sys.stdout.flush(); sys.stderr.flush()
        except Exception:
            pass

        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd); os.close(stderr_fd)

        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

        devnull.close()

        logging.disable(prev_logging_disable)
        if prev_tqdm_disable is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = prev_tqdm_disable

def lse(fast_factor):
    return torch.logsumexp(fast_factor.tensor.reshape(-1), dim=0)

def get_fw_bw_correlation(output_message, mg):
    from nce.inference.factor import FastFactor
    output_message.order_indices()
    mg.order_indices()
    fw_mean = output_message.tensor.mean()
    bw_mean = mg.tensor.mean()
    # compute (fw_message - fw_mean) * (bw_message - bw_mean) (not logspace +)
    # @ does true product of logspace elements
    adjusted_avg_product = torch.mean(
        (FastFactor(output_message.tensor - fw_mean, output_message.labels) @ \
        FastFactor(mg.tensor - bw_mean, mg.labels)).tensor
    )
    fm_std = output_message.tensor.std(unbiased=False)
    mg_std = mg.tensor.std(unbiased=False)
    if fm_std == 0 or mg_std == 0:
        return 0
    return adjusted_avg_product.item() / (output_message.tensor.std(unbiased=False).item() * mg.tensor.std(unbiased=False).item())

def get_message_stats(gm, bucket, output_message, get_Z_estimands=True):
    # adds (label, width, fw_var, bw_var, const_pred_linspace_mse_Z_err, ls_one)
    import os, contextlib
    from .backward_message import get_backward_message
    stats = dict()
    stats['label'] = bucket.label
    stats['width'] = len(output_message.labels)
    if not bucket.approximate_downstream_factors or len(output_message.labels) == 0:
        empty = True
    else:
        empty = False
        with open(os.devnull, "w") as devnull, \
            contextlib.redirect_stdout(devnull), \
            contextlib.redirect_stderr(devnull):
            mg = get_backward_message(bucket.gm, bucket.label, bucket.approximate_downstream_factors)[0]
    
    stats['mg_var'] = 0 if empty or mg is None else mg.get_variance()
    if not get_Z_estimands:
        Z_errs = () # linspace_err,logspace_err
    else:
        Z_linspace_mse_fw_component = torch.logsumexp(output_message.tensor.reshape(-1), dim=0) - math.log(output_message.tensor.numel())
        Z_logspace_mse_bw_component = torch.mean(output_message.tensor.reshape(-1))
        if len(output_message.labels) == 0:
            stats['linspace_err'], stats['logspace_err'] = 0, 0
        elif not bucket.approximate_downstream_factors:
            stats['linspace_err'], stats['logspace_err'] = 0, Z_linspace_mse_fw_component.item() - Z_logspace_mse_bw_component.item()
        else:
            if mg is not None:
                Z = (output_message * mg).sum_all_entries()
                Z_component_from_bw_message = torch.logsumexp(mg.tensor.reshape(-1), dim=0)
            else:
                Z = output_message.sum_all_entries()
                
            fw_star = Z - Z_component_from_bw_message
            stats['linspace_err'], stats['logspace_err'] = \
                Z_linspace_mse_fw_component.item() - fw_star.item(), \
                Z_logspace_mse_bw_component.item() - fw_star.item()
            
    stats['output_message_var'] = output_message.get_variance()
    stats['correlation'] = get_fw_bw_correlation(output_message, mg) if not empty else 0
    stats['output_message_std'] = stats['output_message_var']** 0.5 if stats['output_message_var'] > 0 else 0
    stats['mg_std'] = stats['mg_var'] ** 0.5 if stats['mg_var'] > 0 else 0
    stats['correlation_correction_factor'] = 2 * stats['correlation'] * stats['output_message_std'] * stats['mg_std']
    stats['corrected_err'] = stats['linspace_err'] + stats['correlation_correction_factor']


    gm.message_stats.append(stats)
    return stats


def get_gm_message_stats(gm, ecl):
    from .backward_message import get_backward_message
    with mute_everything():
        gm_copy = copy.deepcopy(gm)
        gm_copy.loss_fn = 'none'
        gm_copy.ecl = 10**30
        gm_copy.iB = 100
        # eliminate all variables one by one and gather stats when ecl is exceeded
        stats = {}
        elim_order = gm_copy.elim_order
        for var in elim_order:
            current_bucket = gm_copy.buckets[var]
            label = current_bucket.label
            if current_bucket.get_ec() > ecl:
                g, f = get_backward_message(gm_copy, label)
                sigma_f = f.tensor.std(unbiased=False)
                sigma_g = g.tensor.std(unbiased=False)
                rho = get_fw_bw_correlation(f, g)
                stats[label] = {
                    'sigma_f': sigma_f,
                    'sigma_g': sigma_g,
                    'rho': rho
                }
            gm_copy.eliminate_variables([var])
        var_g_avg, rho_avg = 0, 0
        for key in stats.keys():
            var_g_avg += stats[key]['sigma_g']**2
            rho_avg += stats[key]['rho']
        var_g_avg /= len(stats)
        rho_avg /= len(stats)
        return var_g_avg.item(), rho_avg, stats
    
def get_permuted_avg_err(f, f_hat, b, num_samples = 100):
    b_copy = copy.deepcopy(b)
    errs = []
    abs_errs = []
    for _ in range(num_samples):
        b_copy.shuffle()
        z = (f * b_copy).sum_all_entries()
        z_hat = (f_hat * b_copy).sum_all_entries()
        err = (z_hat - z)
        errs.append(err)
        abs_errs.append(abs(err))
    return sum(errs) / len(errs), sum(abs_errs) / len(abs_errs)