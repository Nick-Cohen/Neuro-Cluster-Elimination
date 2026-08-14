"""GPU discovery and idleness detection for the rerun scheduler.

DESIGN NOTE -- WHY THERE IS ALMOST NO THRESHOLD HERE
----------------------------------------------------
The obvious way to write this is "a GPU is idle if memory_used < X MiB and
utilisation < Y%". Both X and Y would be invented numbers, and this project has
a standing rule against inventing operational parameters.

So the PRIMARY signal is not a threshold at all, it is a fact:
`nvidia-smi --query-compute-apps` lists every process holding a CUDA context.
Empty list => nobody is computing on that GPU. That is not an estimate.

Utilisation and memory are recorded for the run manifest and used only as a
*secondary* corroborating signal, behind an explicitly-documented default that
was MEASURED on this box rather than guessed (see IDLE_MEMORY_MIB below).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# GPU 2 IS RETIRED. DO NOT REMOVE THIS.
# ---------------------------------------------------------------------------
# GPU 2 (GPU-51c4e7bc-19d8-3b02-459b-8f79fcb99935) is excluded from all
# dispatch because it produced SILENT DATA CORRUPTION: runs completed with
# exit code 0 and returned wrong numbers. There is no error message to detect
# and no exception to catch -- the only defence is never to send work there.
#
# This is a correctness exclusion, not a performance one. A rerun executed on
# GPU 2 is indistinguishable from a good run except that its numbers are wrong,
# which is the worst possible failure mode for a paper rerun.
#
# If you are considering re-enabling it: the burden of proof is a full
# determinism-regression pass ON GPU 2 matching the goldens bit-for-bit, not
# "it looked fine when I tried it". Until then, leave this alone.
RETIRED_GPU_INDICES = frozenset({2})
RETIRED_GPU_UUIDS = frozenset({'GPU-51c4e7bc-19d8-3b02-459b-8f79fcb99935'})

# Secondary corroborating signal only; the primary test is "no compute apps".
# MEASURED on this box 2026-08-14: all four TITAN RTX report exactly 1 MiB used
# and 0% utilisation with no processes attached. The default is set to 64 MiB
# to tolerate driver-side bookkeeping (e.g. a display context) without
# tolerating a real workload, whose smallest footprint here is ~600 MiB (a bare
# torch CUDA context). Override via NCE_SCHED_IDLE_MEMORY_MIB.
IDLE_MEMORY_MIB = int(os.environ.get('NCE_SCHED_IDLE_MEMORY_MIB', '64'))


@dataclass(frozen=True)
class GpuInfo:
    index: int
    uuid: str
    name: str
    memory_used_mib: int
    memory_total_mib: int
    utilization_pct: int
    compute_pids: tuple = field(default=())

    @property
    def retired(self) -> bool:
        return self.index in RETIRED_GPU_INDICES or self.uuid in RETIRED_GPU_UUIDS

    @property
    def busy_reason(self) -> Optional[str]:
        """Return a human-readable reason this GPU is unavailable, else None."""
        if self.retired:
            return 'RETIRED (silent data corruption) -- never dispatch'
        if self.compute_pids:
            return 'compute processes present: %s' % (
                ','.join(str(p) for p in self.compute_pids))
        if self.memory_used_mib > IDLE_MEMORY_MIB:
            return 'memory in use: %d MiB > %d MiB' % (
                self.memory_used_mib, IDLE_MEMORY_MIB)
        return None

    @property
    def idle(self) -> bool:
        return self.busy_reason is None


def _nvidia_smi(query: str, extra: List[str] = None) -> List[List[str]]:
    cmd = ['nvidia-smi', '--query-%s' % query, '--format=csv,noheader,nounits']
    if extra:
        cmd.extend(extra)
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if line:
            rows.append([c.strip() for c in line.split(',')])
    return rows


def query_gpus(ignore_pids: Optional[Iterable[int]] = None) -> List[GpuInfo]:
    """Snapshot every GPU on the box, including retired ones (marked as such).

    Sorted by index so iteration order is deterministic.

    `ignore_pids` exists for thermal ballast (see nce/scheduler/ballast.py).
    Ballast attaches a real CUDA context, so without this the authoritative
    "a compute process is attached" busy signal would report every ballasted
    card as busy and the scheduler would deadlock -- box fully utilised,
    nothing dispatched. ONLY the scheduler's own ballast pids belong here;
    passing anything else would hide real work and let two jobs collide on one
    card.
    """
    ignore = set(ignore_pids or ())
    # Which GPUs have live CUDA contexts? This is the authoritative busy signal.
    pids_by_uuid: Dict[str, List[int]] = {}
    # Memory attributable to ignored (ballast) pids, so the secondary memory
    # signal does not re-flag a card that the pid filter just cleared. A ballast
    # worker holds ~1-2 GiB (context + two 8192^2 fp32 matrices), far above
    # IDLE_MEMORY_MIB, so skipping this would defeat the pid filter entirely.
    ignored_mib_by_uuid: Dict[str, int] = {}
    for row in _nvidia_smi('compute-apps=gpu_uuid,pid,used_gpu_memory'):
        if len(row) >= 2:
            pid = int(row[1])
            if pid in ignore:
                if len(row) >= 3:
                    try:
                        ignored_mib_by_uuid[row[0]] = (
                            ignored_mib_by_uuid.get(row[0], 0) + int(row[2]))
                    except ValueError:
                        pass
                continue
            pids_by_uuid.setdefault(row[0], []).append(pid)

    gpus = []
    for row in _nvidia_smi(
            'gpu=index,uuid,name,memory.used,memory.total,utilization.gpu'):
        idx, uuid, name, mused, mtotal, util = row[:6]
        used = max(0, int(mused) - ignored_mib_by_uuid.get(uuid, 0))
        gpus.append(GpuInfo(
            index=int(idx),
            uuid=uuid,
            name=name,
            memory_used_mib=used,
            memory_total_mib=int(mtotal),
            utilization_pct=int(util),
            compute_pids=tuple(sorted(pids_by_uuid.get(uuid, []))),
        ))
    return sorted(gpus, key=lambda g: g.index)


def dispatchable_gpus(ignore_pids: Optional[Iterable[int]] = None) -> List[GpuInfo]:
    """GPUs that are safe to dispatch to, in deterministic (index) order.

    Retired GPUs are filtered out here as well as in `busy_reason`, so a caller
    that forgets to check `.retired` still cannot be handed GPU 2.

    `ignore_pids`: see `query_gpus`. A card carrying only our thermal ballast is
    dispatchable -- the scheduler tears the ballast down before it launches.
    """
    return [g for g in query_gpus(ignore_pids=ignore_pids)
            if not g.retired and g.idle]


def assert_not_retired(index: int) -> None:
    """Hard guard for any code path that takes an explicit device index."""
    if index in RETIRED_GPU_INDICES:
        raise RuntimeError(
            'Refusing to use cuda:%d -- GPU %d is RETIRED for silent data '
            'corruption. See RETIRED_GPU_INDICES in nce/scheduler/gpus.py.'
            % (index, index))


def describe() -> str:
    """One-line-per-GPU human summary, for logs and --status."""
    lines = []
    for g in query_gpus():
        state = 'IDLE' if g.idle else ('BUSY: %s' % g.busy_reason)
        lines.append('  cuda:%d %-18s %5d/%5d MiB %3d%%  %s  [%s]'
                     % (g.index, g.name, g.memory_used_mib, g.memory_total_mib,
                        g.utilization_pct, state, g.uuid))
    return '\n'.join(lines)
