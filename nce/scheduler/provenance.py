"""Run provenance: everything needed to reconstruct a run later.

Priority-1 module. Cheap to record, and every subsequent run needs it, so it is
deliberately independent of the scheduler -- you can call `capture()` from a
plain script with no queue involved.

WHAT IS RECORDED AND WHY EACH ITEM IS HERE
------------------------------------------
seeds            The config seed, plus the derived per-bucket seed formula, so
                 a reader can recompute any bucket's RNG anchor.
git              Exact commit, branch, and whether the tree was DIRTY. A commit
                 alone is a lie if the tree had uncommitted edits, which in this
                 repo (many concurrent worktrees) is the normal case.
resolved_config  The config AFTER `prepare_config` fills schema defaults -- not
                 the input dict. Defaults change over time; the input dict does
                 not tell you what actually ran.
hardware         GPU **UUID**, not index. CUDA_VISIBLE_DEVICES remaps indices,
                 so "cuda:0" in two logs can be two different cards. The UUID is
                 the only stable identity.
libraries        torch / numpy / CUDA / cuDNN / driver versions.
threads          torch thread count. MEASURED in the determinism suite: the same
                 CPU case returns values 2 float32 ULP apart at 1 vs 4 threads.
                 A run without this recorded is not reproducible.
timing           Sample-generation vs NN-training, split. See timing.py for why
                 this split specifically is load-bearing.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

SCHEMA_VERSION = 1


def _git(args, cwd) -> Optional[str]:
    try:
        out = subprocess.run(['git'] + args, cwd=cwd, capture_output=True,
                             text=True, timeout=30)
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:
        return None


def git_provenance(cwd: str = None) -> Dict[str, Any]:
    """Commit, branch, dirty flag and the diff stat of the working tree.

    `dirty` matters more than `commit` in this repo: with a dozen live
    worktrees, uncommitted edits are normal, and a manifest that reports only
    the commit would silently misdescribe the code that ran.
    """
    cwd = cwd or os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    status = _git(['status', '--porcelain'], cwd)
    return {
        'repo_root': cwd,
        'commit': _git(['rev-parse', 'HEAD'], cwd),
        'commit_short': _git(['rev-parse', '--short', 'HEAD'], cwd),
        'branch': _git(['rev-parse', '--abbrev-ref', 'HEAD'], cwd),
        'describe': _git(['describe', '--always', '--dirty'], cwd),
        'dirty': bool(status),
        'dirty_files': sorted(
            l[3:] for l in status.splitlines()) if status else [],
        'commit_subject': _git(['log', '-1', '--format=%s'], cwd),
        'commit_date': _git(['log', '-1', '--format=%cI'], cwd),
    }


def library_provenance() -> Dict[str, Any]:
    info = {
        'python': sys.version.split()[0],
        'python_executable': sys.executable,
        'platform': platform.platform(),
    }
    try:
        import torch
        info['torch'] = torch.__version__
        info['torch_cuda'] = torch.version.cuda
        info['torch_cudnn'] = (torch.backends.cudnn.version()
                               if torch.backends.cudnn.is_available() else None)
        info['torch_num_threads'] = torch.get_num_threads()
        info['cuda_available'] = torch.cuda.is_available()
    except Exception as e:  # pragma: no cover - torch is a hard dep in practice
        info['torch_import_error'] = str(e)
    try:
        import numpy
        info['numpy'] = numpy.__version__
    except Exception:
        pass
    return info


def hardware_provenance(device: str = None) -> Dict[str, Any]:
    """Hardware identity. GPU **UUID** is the point of this function."""
    info = {
        'hostname': socket.gethostname(),
        'cpu_count': os.cpu_count(),
        'device_requested': device,
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
    }
    try:
        from nce.scheduler.gpus import query_gpus
        gpus = query_gpus()
        info['gpus_visible'] = [
            {'index': g.index, 'uuid': g.uuid, 'name': g.name,
             'memory_total_mib': g.memory_total_mib} for g in gpus]
    except Exception as e:
        info['gpu_query_error'] = str(e)

    # Resolve the *actual* card this process will compute on, by UUID.
    try:
        import torch
        if torch.cuda.is_available():
            local_idx = torch.cuda.current_device()
            info['torch_current_device_index'] = local_idx
            info['torch_current_device_name'] = torch.cuda.get_device_name(local_idx)
            cap = torch.cuda.get_device_capability(local_idx)
            info['torch_current_device_capability'] = '%d.%d' % cap
            # CUDA_VISIBLE_DEVICES remaps: torch index i == i-th visible GPU.
            vis = os.environ.get('CUDA_VISIBLE_DEVICES')
            gl = info.get('gpus_visible', [])
            if vis:
                wanted = [t.strip() for t in vis.split(',') if t.strip()]
                if local_idx < len(wanted):
                    tok = wanted[local_idx]
                    match = [g for g in gl
                             if g['uuid'] == tok or str(g['index']) == tok]
                    if match:
                        info['gpu_uuid_in_use'] = match[0]['uuid']
                        info['gpu_physical_index'] = match[0]['index']
            elif local_idx < len(gl):
                info['gpu_uuid_in_use'] = gl[local_idx]['uuid']
                info['gpu_physical_index'] = gl[local_idx]['index']
    except Exception as e:
        info['torch_device_error'] = str(e)
    return info


def seed_provenance(resolved_config: Dict[str, Any]) -> Dict[str, Any]:
    """The seed, plus the formula that turns it into per-bucket RNG anchors.

    Recorded as a formula string rather than a list of values because the set of
    buckets is not known until the elimination order is computed, and because
    the formula is the thing a future reader needs in order to reproduce any
    single bucket in isolation.
    """
    return {
        'config_seed': resolved_config.get('seed'),
        'dt_random_seed': resolved_config.get('dt_random_seed'),
        # nce/sampling/sample_generator.py::_get_seed
        'sample_generator_seed_formula':
            'bucket_label + config_seed*10000 + counter*100 '
            '+ (50000000 if validation else 0)',
        # nce/neural_networks/net.py::Net.__init__ re-anchors before init
        'nn_init_seed': 'torch.manual_seed(config_seed) at each Net construction',
        'rng_is_reanchored_per_bucket': True,
    }


@dataclass
class RunManifest:
    """The complete provenance record for one run. Serialises to JSON."""
    run_id: str
    schema_version: int = SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    job: Dict[str, Any] = field(default_factory=dict)
    resolved_config: Dict[str, Any] = field(default_factory=dict)
    seeds: Dict[str, Any] = field(default_factory=dict)
    git: Dict[str, Any] = field(default_factory=dict)
    libraries: Dict[str, Any] = field(default_factory=dict)
    hardware: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    status: str = 'running'

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)

    def write(self, path: str) -> None:
        """Atomic write -- a manifest half-flushed by a kill is worse than none."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w') as fh:
            fh.write(self.to_json())
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    @staticmethod
    def read(path: str) -> Dict[str, Any]:
        with open(path) as fh:
            return json.load(fh)


# Environment variables that demonstrably change numerics or performance.
# Anything that can move a result belongs here; anything that cannot, does not.
_ENV_KEYS_OF_INTEREST = (
    'CUDA_VISIBLE_DEVICES',
    'CUBLAS_WORKSPACE_CONFIG',   # required by use_deterministic_algorithms
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
    'PYTORCH_CUDA_ALLOC_CONF',
    'NCE_SCHED_IDLE_MEMORY_MIB',
)


def capture(run_id: str,
            resolved_config: Dict[str, Any],
            job: Dict[str, Any] = None,
            model_info: Dict[str, Any] = None,
            repo_root: str = None) -> RunManifest:
    """Build a full manifest. `resolved_config` must be POST-`prepare_config`."""
    return RunManifest(
        run_id=run_id,
        job=dict(job or {}),
        resolved_config=dict(resolved_config),
        seeds=seed_provenance(resolved_config),
        git=git_provenance(repo_root),
        libraries=library_provenance(),
        hardware=hardware_provenance(resolved_config.get('device')),
        model=dict(model_info or {}),
        env={k: os.environ.get(k) for k in _ENV_KEYS_OF_INTEREST},
    )
