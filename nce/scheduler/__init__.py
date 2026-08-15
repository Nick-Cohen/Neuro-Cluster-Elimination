"""GPU job scheduling, run provenance, and checkpoint/resume for paper reruns.

Built so idle GPUs automatically take queued rerun work instead of sitting at
0% while decisions are pending.

Quick start
-----------
    # 1. build a queue (deterministic job ids; safe to re-run)
    python -m nce.scheduler.enqueue --queue Q.json \
        --problems grids/grid10x10.f10 pedigree/pedigree1 \
        --strategies reduce_nn nomerge --bounds 4 8 --seeds 42 43 \
        --base-config base.json

    # 2. see what it would do
    python -m nce.scheduler.scheduler --queue Q.json --status

    # 3. run it (fills every genuinely idle GPU, never GPU 2)
    python -m nce.scheduler.scheduler --queue Q.json --out-dir runs/

Modules
-------
gpus         idle detection + the hard GPU-2 exclusion
jobs         JobSpec / JobQueue, content-addressed deterministic ids
models       .model_cache validation (catches the HTML-error-page case)
provenance   RunManifest: seeds, git, resolved config, GPU UUID, libs, timing
timing       sample-generation vs NN-training split, per problem and network
checkpoint   cluster-level message journal; bit-identical resume
runner       executes one job in its own process (pins CUDA_VISIBLE_DEVICES)
reaper       returns jobs stuck at `running` (dead scheduler) to the queue
scheduler    the dispatch loop
"""

from nce.scheduler.gpus import (GpuInfo, query_gpus, dispatchable_gpus,
                                assert_not_retired, describe,
                                RETIRED_GPU_INDICES)
from nce.scheduler.jobs import JobSpec, JobQueue, build_grid, NEEDS_ATTENTION
# NOTE: `reaper` is deliberately NOT imported here. It is a __main__ entry point
# (`python -m nce.scheduler.reaper`), and importing it from the package __init__
# makes runpy load it twice -- once as `nce.scheduler.reaper` via this line and
# once as `__main__` -- which it warns about. Import it by module path.
from nce.scheduler.provenance import RunManifest, capture
from nce.scheduler.timing import collect_timings, TimingCollector
from nce.scheduler.checkpoint import (CheckpointStore, checkpointed_elimination,
                                      journal_fingerprint, verify_environment)
from nce.scheduler import models

__all__ = [
    'GpuInfo', 'query_gpus', 'dispatchable_gpus', 'assert_not_retired',
    'describe', 'RETIRED_GPU_INDICES',
    'JobSpec', 'JobQueue', 'build_grid', 'NEEDS_ATTENTION',
    'RunManifest', 'capture',
    'collect_timings', 'TimingCollector',
    'CheckpointStore', 'checkpointed_elimination', 'journal_fingerprint',
    'verify_environment', 'models',
]
