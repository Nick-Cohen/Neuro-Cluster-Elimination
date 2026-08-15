# 61 — the corrected paper rerun, phase 1: launched

**Date:** 2026-08-14 · **Build:** `frozen-rerun-v3` (`5d3c8d4`, branch `frozen/rerun-v1`)
· **Worktree:** `/tmp/claude-58902/wt-frozen` · **Output:** `/home/cohenn1/NCE-rerun/phase1/`

> ## IS IT RUNNING? YES.
>
> | | |
> |---|---|
> | **queued** | **1392 jobs** = 29 benchmark cells × 16 arms × 3 seeds (42/43/44) |
> | **arms** | `nomerge`, `nonsub12`, `rnn{2,4,6,8,10,12,14}`, `sub{2,4,6,8,10,12,14}` |
> | **cards** | **gpu0 + gpu3 only.** gpu1 not used at all. gpu2 never touched. |
> | **expected** | **647 GPU-h** by the original study's own wall times → **~13.5 days** on two cards. Longer in practice: the sample-count fix raises sample counts. |
> | **order** | **shortest-job-first**, from the study's measured `time_min` |
> | **thermal** | ballast on, `--warmup-s 360`; no job in the sweep starts cold |
> | **scheduler** | pid in `/home/cohenn1/NCE-rerun/phase1/scheduler.pid`, log `scheduler.log` |
>
> **Evidence it is producing correct results, not just running:**
> the 12 exact-path cells (`rnn{6,8,10,12,14}` on `grid10x10.f10`, `num_trained=0`)
> return `303.08575439453125` against an exact solve of `303.0858154296875` —
> **6.1e-5, i.e. float32 round-off** — identically on both cards and all three
> seeds. The NN-path cells land on the original study's own numbers:
>
> | arm | rerun err (mean ± sd, 3 seeds) | study err | rerun #NN | study #NN | rerun min | study min |
> |---|---|---|---|---|---|---|
> | `rnn4` | 0.192 ± 0.087 | 0.247 ± 0.239 | 1 | 1 | 0.46 | 0.4 |
> | `sub4` | 0.078 ± 0.029 | 0.038 ± 0.014 | 1 | 1 | 0.42 | 0.4 |
> | `rnn6/8/12` | 0.0001 | 0.0002 | 0 | 0 | 0.11 | 0.0 |
>
> Exact agreement is **not** expected and would be a red flag: CRN default-on
> deliberately moved every sampled number (doc 60 §7). Structure (#NN), timing and
> error magnitude reproducing is exactly the right signature.
>
> **Resume was re-tested on this build and is bit-identical.** A job SIGKILLed at
> 60 s resumed with **86 clusters replayed / 14 computed** and returned
> `302.9474182128906` — the same `repr` as the from-scratch run, with identical
> per-bucket epoch counts `[332, 228, 135, 312, 284, 452]`.

---

## 1. What phase 1 is, and where every number came from

Phase 1 is the paper's benchmark set at merge bounds **through `e_max` 14**, at
**3 seeds**. Nothing was invented:

| input | source |
|---|---|
| the 29 cells, `iB`, `ecl = 2**iB + 1`, reference log Z | `notebooks/June-2026/claude_experiments/reduce_nn_experiment/benchmark_set.json` (the `ecl` rule is re-asserted per group at enqueue time) |
| the config body of every arm | the study's own per-arm YAML, `reduce_nn_experiment/configs/*.yaml` |
| the merge flags per arm | the same YAMLs (see §2) |
| dispatch order | the study's measured `time_min` in `results_for_writeup/results_table.csv` |

**Config fidelity was checked, not assumed.** `61_enqueue_phase1.py --verify-configs`
diffs the config it would hand to `FastGM` against every study YAML it can pair
with, field for field, in both directions (missing *and* extra keys), and refuses
to enqueue on any mismatch:

```
config fidelity: 1107 YAML(s) paired and checked, 0 field mismatch(es),
                 852 study YAML(s) outside phase 1 (ignored)
```

The remaining 285 phase-1 jobs are arms the original study never ran
(`rnn10`/`rnn14` on some cells; `rbm_20` and `rbm_ferro_20` at iB20), so there is
no YAML to pair with — they go through the identical code path.

Every model was validated **by content**, not existence — 26 distinct problems,
0 bad — before a single job was queued. `NCE_MODEL_CACHE=/home/cohenn1/NCE/.model_cache`
is set for the scheduler and inherited by every runner; the build asserts it at
startup in `enqueue`, `scheduler` and `runner` alike.

## 2. One change to the frozen build was necessary — and it is orchestration-only

**`frozen-rerun-v2` could not express 14 of the 16 arms.** `JobSpec.to_config`
knew `reduce_nn`, `nomerge` and `merge_degree` and nothing else. Three defects,
all in `nce/scheduler/jobs.py`:

| | |
|---|---|
| `subsumption` (`use_join_tree_merge`) and `non_subsumption` (`use_non_subsumption_merge`) | **no mapping at all** — the `sub*` and `nonsub12` arms were unrepresentable |
| `reduce_nn` | **did not set `reduce_nn_backtrack`**, which every `rnn` YAML in the study sets. Without it `reduce_nn_merge` stops at the first bound reaching the minimum NN count instead of binary-searching down to the smallest one — a structurally different merge, i.e. an arm that is not the paper's |
| `merge_degree` | set only `max_merge_bound`, which no merge pass reads (every pass is gated on its own `use_*_merge` flag), making it a silent duplicate of `nomerge`. Not used by the rerun; fixed so it cannot mislead later |

Fixed in `5d3c8d4`, tagged **`frozen-rerun-v3`**. This is the same scope
`frozen-rerun-v2` took over v1 — orchestration only:

```
$ git diff --name-only frozen-rerun-v2 frozen-rerun-v3 -- nce/
nce/scheduler/jobs.py
$ git diff --stat frozen-rerun-v2 frozen-rerun-v3 -- nce/inference nce/sampling \
      nce/neural_networks nce/config_schema.py
(empty)
```

**Every number in doc 60 stands.** `tests/test_scheduler_dispatch.py`,
`test_scheduler_resume.py` and `test_ballast.py`: **39 passed, 1 skipped**.

## 3. GPU policy: gpu0 + gpu3 only, and why

**Timing is a reported quantity for every cell** — the study's table has one
`time_min` per (problem, arm), averaged over its seeds. gpu1 is thermally
throttled ~10% and sustained; its *numbers* are fine but its *timings* are not
comparable. Putting one seed of three on gpu1 would push a **~3.3% systematic
inflation into every reported time**, invisible in the CSV, and would have to be
carried as a caveat through the whole writeup. Two comparable cards beat three
incomparable ones.

So: `--only-gpus 0 3`. Card assignment is going out balanced on its own — **11
jobs on gpu0, 11 on gpu3** in the first 22 — which matters because doc 59
measured gpu0 and gpu3 differing by 1.44% at steady state.

The cost is throughput: ~13.5 days rather than ~9. **If Nick would rather have
gpu1's throughput than timing comparability, say so and I will re-partition** —
the queue is content-addressed, so adding a card costs nothing and repeats no
finished work.

Also fixed by policy, not preference: `--threads 1` (the determinism suite
measured a 2-float32-ULP shift between 1 and 4), ballast on by default, and
`--warmup-s 360`, which closes the only remaining cold-start case (the first job
on each card). Measured residual thermal bias with ballast handoff: **+0.219%**,
against a **−5.25%** cold-start effect.

## 4. Pre-flight, as run

- `nvidia-smi`: all four cards at 1 MiB / 0%. `ps aux | grep python`: no
  non-system processes. No orphans from earlier today.
- 1392 specs → **1392 distinct `job_id`s** (BLAKE2b of the canonical spec; a
  collision would silently overwrite results).
- Dry run printed the job list, the count, the estimate provenance and the
  GPU-hour total **before** anything was written.
- Time estimates: 363 cells exact from the study's table; the other 101 filled by
  `nearest` bound within the same arm family (61), `cellmean` (8), or `analogue`
  sibling cell (32). Every row's rule is recorded in
  `61-phase1-estimates.csv`. **This affects dispatch order only and never a config.**
- Total reproduces the brief's figure: the arms' `time_min × n_seeds` over the
  cells the study covered sums to **564 GPU-h**, and 647 GPU-h with the cells it
  did not.

## 5. Evidence the pipeline works end to end

```
[warmup] holding 2 card(s) at load for 360s before the first dispatch: cuda:0,cuda:3
[warmup] done
[dispatch] cuda:0 <- grids_grid10x10.f10__reduce_nn6__s42__b6eff2a0169fef5f (pid 3896770)
[dispatch] cuda:3 <- grids_grid10x10.f10__reduce_nn6__s43__daa6887e52c56ae5 (pid 3896774)
[done]     grids_grid10x10.f10__reduce_nn6__s42__b6eff2a0169fef5f after 30.4s on cuda:0
[done]     grids_grid10x10.f10__reduce_nn6__s43__daa6887e52c56ae5 after 30.4s on cuda:3
[dispatch] cuda:0 <- grids_grid10x10.f10__reduce_nn6__s44__76b65d8cb7ad2303 (pid 3897212)
[dispatch] cuda:3 <- grids_grid10x10.f10__reduce_nn8__s42__624a6d3140f44b6c (pid 3897213)
```

dispatch → run → manifest → result → **freed card picked up automatically on the
next sweep**, on both cards, in shortest-job-first order. Provenance on all 22
completed manifests: `git.commit = 5d3c8d4a`, `git.dirty = False`, and a recorded
`gpu_uuid_in_use` / `gpu_physical_index`.

Read timings from `manifest.timing.t_elim_s` / `total_wall_s`, **never** from the
`[done] … after Xs` console line — that is quantised to the 30 s poll interval
and overstates duration (doc 60 §6.3). The line above says 30.4 s; the manifest
says 6.3 s.

**Resume** (`/home/cohenn1/NCE-rerun/validation/`): `grid10x10.f10 nomerge s42`
run to completion (`302.9474182128906`, 6 NNs, 128 s), then run again and SIGKILLed
at 60 s, then restarted into the same output directory:

| | from scratch | after kill + resume |
|---|---|---|
| clusters replayed / computed | 0 / 100 | **86 / 14** |
| log Z (`repr`) | `302.9474182128906` | **`302.9474182128906`** |
| per-bucket epochs | `[332, 228, 135, 312, 284, 452]` | **identical** |
| wall | 128 s | 81 s |

Doc 55's bit-identical-resume property survives into this build. The SIGKILL left
no orphan and no CUDA context behind.

That job's log Z is also a sanity check in its own right: `abs_err = 0.138`
against the exact solve, versus the study's `nomerge` mean of `0.567 ± 0.211`,
and `num_trained = 6` against the study's `num_NN = 6`.

---

## 6. How Nick monitors it

```bash
export NCE_MODEL_CACHE=/home/cohenn1/NCE/.model_cache
cd /tmp/claude-58902/wt-frozen

# progress + accuracy so far, against exact_logZ.json / benchmark_set.json refs
python notebooks/_August-2026/claude_experiments/61_status.py \
    --csv /home/cohenn1/NCE-rerun/phase1/status_snapshot.csv

# queue + card state
python -m nce.scheduler.scheduler --queue /home/cohenn1/NCE-rerun/phase1/queue.json --status

# live dispatch log
tail -f /home/cohenn1/NCE-rerun/phase1/scheduler.log
```

`61_status.py` reads only files on disk, so it is safe to run against a live
sweep. It reports non-finite log Z counts and the worst-error cells first — that
is the thing to glance at.

## 7. How to resume it if it dies

The queue is content-addressed and every job journals per cluster, so restarting
is safe and repeats no finished work.

```bash
# 1. is it actually dead?
ps -p "$(cat /home/cohenn1/NCE-rerun/phase1/scheduler.pid)" || echo dead

# 2. IMPORTANT: jobs the dead scheduler had in flight are stuck at 'running' and
#    will never be re-dispatched. Put them back:
python - <<'PY'
import json
p='/home/cohenn1/NCE-rerun/phase1/queue.json'
s=json.load(open(p)); n=0
for r in s['jobs'].values():
    if r['status'] in ('running','failed'):
        r['status']='pending'; n+=1
json.dump(s, open(p,'w'), indent=2, sort_keys=True); print('reset',n)
PY

# 3. relaunch. Its journal makes each interrupted job resume mid-elimination.
cd /tmp/claude-58902/wt-frozen
export NCE_MODEL_CACHE=/home/cohenn1/NCE/.model_cache
setsid nohup python -u -m nce.scheduler.scheduler \
    --queue /home/cohenn1/NCE-rerun/phase1/queue.json \
    --out-dir /home/cohenn1/NCE-rerun/phase1/runs \
    --threads 1 --only-gpus 0 3 --warmup-s 360 \
    >> /home/cohenn1/NCE-rerun/phase1/scheduler.log 2>&1 < /dev/null &
```

Step 2 is the only non-obvious part and it is the one that bites: the scheduler
has no stale-`running` reaper, so without it those jobs sit in the queue as
permanently in-flight.

To **add work** (phase 2 at higher `e_max`, or the 10-seed extension) just re-run
the enqueue script with wider `BOUNDS`/`SEEDS` — `add()` is idempotent by
`job_id` and leaves completed work alone. To **add gpu1 for accuracy-only work**,
run a *second* scheduler against a *separate* queue with `--only-gpus 1` and no
`--warmup-s`; do not add it to this one, or its timings will silently enter the
paper's numbers.

## 8. Artefacts

| path | what |
|---|---|
| `/home/cohenn1/NCE-rerun/phase1/queue.json` | the 1392-job queue |
| `/home/cohenn1/NCE-rerun/phase1/runs/<job>/` | `manifest.json`, `result.json`, `runner.log`, `checkpoint/` |
| `/home/cohenn1/NCE-rerun/phase1/scheduler.log`, `.pid` | dispatch log, scheduler pid |
| `/home/cohenn1/NCE-rerun/validation/` | the resume + log Z validation runs of §5 |
| `61_enqueue_phase1.py` | queue builder, SJF ordering, `--verify-configs` |
| `61-phase1-estimates.csv` | per-(cell, arm) time estimate + the rule used |
| `61_status.py` | progress + accuracy aggregator |

All scripts are committed on `frozen/rerun-v1` (`5d3c8d4`, `5710395`).
