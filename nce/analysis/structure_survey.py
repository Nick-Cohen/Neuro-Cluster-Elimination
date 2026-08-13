"""Build-only structural survey of NCE bucket/cluster structure.

WHY THIS EXISTS
---------------
Several optimisation questions ("is it worth sharing the one-hot encoding
between co-resident NN factors?", "does an NN get re-evaluated redundantly
across streamed blocks?", "how big will this cluster's sample-gen grid be?")
are answered entirely by the *structure* of the post-merge bucket tree.
Answering them by running inference costs GPU-hours; answering them here
costs milliseconds per problem.

WHERE IN THE LIFECYCLE WE MEASURE  (read this before trusting any number)
------------------------------------------------------------------------
Bucket state changes during elimination.  The quantities that matter
(cluster scope, number of *co-resident NN factors*, elimination grid size)
are properties of the merged, mid-elimination structure, NOT of the initial
factor placement, and NOT of the static join-tree message scopes.

So this module measures each cluster **at the moment ``process_bucket`` is
about to be called on it** -- i.e. after every earlier cluster in the
elimination order has already delivered its message into it.  It gets there
by replaying ``FastGM.eliminate_variables``' bookkeeping *exactly*:

  * the merge passes are the real ones (they run inside ``FastGM.__init__``);
  * the exact/NN dispatch is the real predicate from ``process_bucket``;
  * routing uses the real ``FastGM.find_next_bucket``;
  * delivery uses the real ``FastBucket.receive_message``;
  * cluster scope/width/complexity use the real ``FastBucket`` accessors,
    which read ``bucket.factors``.

The only thing replaced is the *numeric content* of a message: instead of a
tensor (or a trained ``FactorNN``) we deliver a ``_StubFactor`` carrying the
message's label set and an ``is_nn`` flag.  Every structural accessor used
above reads only ``factor.labels`` and ``factor.get_factor_complexity()``,
both of which the stub reproduces exactly (a dense message's ``numel`` is
the product of its domain sizes, which is what the stub returns).

Consequence: ``survey_gm`` CONSUMES the FastGM (it empties ``gm.buckets``,
just as a real elimination does).  Build a fresh one per survey.

THE STREAMING / CHUNKING SITES  (established by reading the source)
------------------------------------------------------------------
There is no single "chunked streaming" switch.  There are five distinct
chunk/stream sites; only two of them ever have *several NN factors alive at
once*, and those are the only two where sharing an encoding between NN
factors, or caching an NN's output across blocks, could possibly pay:

  S1  ``SampleGenerator.sample_tensor_product_elimination`` LARGE path
      trigger: ``elim_prod = prod(domains of the cluster's elim vars) > 2**18``
      Streams over blocks of the elimination grid.  For every block it
      evaluates *every* factor of the cluster -- including every co-resident
      ``FactorNN`` -- at the SAME ``(assignments, elim_coords)`` pair via
      ``FactorNN._eval_elim_block``.  Runs during NN training of that cluster.
      => the natural home of a shared encoding, and of a cross-block cache.

  S2  same function, SMALL path (``elim_prod <= 2**18``): chunks over
      assignments only and calls ``FactorNN._get_slices``, which already
      enumerates ONLY the elim vars present in that factor's scope.

  S3  ``FactorNN._get_slices`` internal batching, ``MAX_QUERY_ROWS = 65536``
      split over assignments.  Single network; no co-residency.

  S4  ``FastBucket.compute_message_exact`` -> ``_compute_message_exact_chunked``
      trigger: ``joint_numel = prod(domains of the cluster scope) > 2**28``
      (``FastBucket._EXACT_JOINT_NUMEL_LIMIT``).  Only reachable on the
      EXACT branch of ``process_bucket``.  With ``stream_nn_exact=True`` each
      co-resident NN factor is re-evaluated per block by
      ``_nn_factor_slice`` -> ``_eval_elim_block``.

  S5  ``FactorNN.nn_to_FastFactor``, ``MAX_QUERY_ROWS = 65536`` over the
      network's own assignment grid.  One network densified once; there is
      nothing to share and nothing to cache.

CROSS-BLOCK REEVALUATION (the S1 / S4 waste this module quantifies)
-------------------------------------------------------------------
S1: ``_eval_elim_block`` is handed the coordinates of ALL of the cluster's
elimination variables.  A factor ``f`` only *reads* the elim variables in
its own scope, ``E_f = labels(f) & elim_labels``.  So as the block loop
sweeps the full ``elim_prod`` grid, ``f``'s value repeats
``elim_prod / prod(dom(E_f))`` times.  Note the dense path (S2,
``_get_slices``) already avoids this -- it enumerates only the present elim
vars -- so this is an optimisation the streaming path is missing, not a new
idea.

S4: the block loop fixes ``batch_labels + batch_elim``; ``f`` only reads the
fixed vars in its own scope, so its slice repeats
``n_blocks / prod(dom(batch & labels(f)))`` times.

Both redundancy factors are computed exactly below.

USAGE
-----
    from nce.analysis import survey_problem, MERGE_STRATEGIES

    s = survey_problem('grids/grid20x20.f10', iB=10, ecl=1025,
                       strategy='reduce_nn', max_merge_bound=16)
    print(s.summary())
    for c in s.clusters:
        ...

or, if you already have a (freshly built, un-eliminated) FastGM:

    from nce.analysis import survey_gm
    s = survey_gm(gm)            # consumes gm
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from nce.inference.factor import FastFactor
from nce.inference.bucket import FastBucket

# --- the real triggers, imported rather than copied where possible ----------
EXACT_JOINT_NUMEL_LIMIT = FastBucket._EXACT_JOINT_NUMEL_LIMIT       # 2**28, S4
EXACT_BLOCK_LIMIT = 2 ** 26          # default block_limit of _compute_message_exact_chunked
SAMPLEGEN_DENSE_LIMIT = 2 ** 18      # elim_prod above which sample-gen streams (S1)
MAX_QUERY_ROWS = 65536               # S3 / S5


MERGE_STRATEGIES = {
    'nomerge':        {},
    'subsumption':    {'use_join_tree_merge': True},
    'reduce_nn':      {'use_reduce_nn_merge': True, 'reduce_nn_backtrack': True},
    'nonsubsumption': {'use_non_subsumption_merge': True},
}


class _StubFactor(FastFactor):
    """A message with no tensor: carries only what the structural accessors read.

    ``FastBucket.get_message_scope`` / ``get_width`` / ``get_ec`` read
    ``factor.labels``; ``get_message_complexity`` reads
    ``factor.get_factor_complexity()``.  For a real dense message the latter is
    ``tensor.numel()`` == product of its domain sizes; for a real ``FactorNN``
    it is also the product of its domain sizes (``FactorNN.get_factor_complexity``).
    So one stub reproduces both faithfully.
    """

    def __init__(self, labels, is_nn, dom):
        super().__init__(None, list(labels))
        self.is_nn = bool(is_nn)
        self._complexity = 1
        for l in self.labels:
            self._complexity *= int(dom[l])

    def get_factor_complexity(self):
        return self._complexity

    def order_indices(self):
        pass


def _prod(labels, dom):
    p = 1
    for l in labels:
        p *= int(dom[l])
    return p


# ---------------------------------------------------------------------------
# S4 block planner -- MIRRORS FastBucket._compute_message_exact_chunked
# ---------------------------------------------------------------------------
def plan_exact_blocks(scope, elim_labels, dom, block_limit=EXACT_BLOCK_LIMIT):
    """Return (batch_labels, batch_elim, n_blocks) for the chunked exact path.

    This is a line-for-line mirror of the batch-variable selection in
    ``FastBucket._compute_message_exact_chunked`` (nce/inference/bucket.py).
    It is duplicated rather than imported because the original is interleaved
    with the actual computation.  ``scope`` must be built the same way the
    original builds it -- ``sorted(union of factor labels)`` -- because the
    selection sort is only stable-deterministic given that order.
    """
    elim_set = set(elim_labels)
    out_labels = [l for l in scope if l not in elim_set]

    remaining = _prod(scope, dom)
    batch_labels = []
    for lbl in sorted(out_labels, key=lambda x: -dom[x]):
        if remaining <= block_limit:
            break
        batch_labels.append(lbl)
        remaining //= int(dom[lbl])

    batch_elim = []
    for lbl in sorted(elim_labels, key=lambda x: -dom[x]):
        if remaining <= block_limit:
            break
        batch_elim.append(lbl)
        remaining //= int(dom[lbl])

    n_blocks = _prod(batch_labels + batch_elim, dom)
    return batch_labels, batch_elim, n_blocks


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
@dataclass
class NNFactorRecord:
    """One co-resident NN factor inside one cluster, with its reeval accounting."""
    labels: List[int]
    width: int
    # --- S1 (sample-gen streaming over the elim grid) ---
    s1_elim_vars_in_scope: int = 0
    s1_distinct_points: int = 1        # prod dom(E_f); the cache would hold this many
    s1_redundancy: float = 1.0         # elim_prod / s1_distinct_points
    # --- S4 (chunked exact message) ---
    s4_free_labels: List[int] = field(default_factory=list)
    s4_rows_per_eval: int = 1          # prod dom(labels \ batch)
    s4_distinct_blocks: int = 1        # prod dom(labels & batch)
    s4_redundancy: float = 1.0         # n_blocks / s4_distinct_blocks


@dataclass
class ClusterRecord:
    """One cluster, measured immediately before ``process_bucket`` runs on it."""
    key_label: int
    elim_labels: List[int]
    message_scope: List[int]
    scope: List[int]                   # union of factor labels == elim + message scope
    n_factors: int
    n_nn_factors: int
    path: str                          # 'exact' | 'nn' | other approximation_method
    width: int                         # len(message_scope)
    message_size: float                # prod dom(message_scope)  == get_ec()
    elim_prod: int                     # prod dom(elim_labels)
    joint_numel: int                   # prod dom(scope)

    # S1: sample-generation streaming (only meaningful when path == 'nn')
    s1_streams: bool = False
    s1_n_blocks: int = 0               # number of elim-grid points swept per assignment

    # S4: chunked exact message (only meaningful when path == 'exact')
    s4_streams: bool = False
    s4_batch_labels: List[int] = field(default_factory=list)
    s4_batch_elim: List[int] = field(default_factory=list)
    s4_n_blocks: int = 0

    nn_factors: List[NNFactorRecord] = field(default_factory=list)

    # --- the two joint conditions the survey exists to answer -------------
    @property
    def s1_joint(self) -> bool:
        """S1 streaming AND >=2 co-resident NN factors."""
        return self.s1_streams and self.n_nn_factors >= 2

    @property
    def s4_joint(self) -> bool:
        """S4 streaming AND >=2 co-resident NN factors."""
        return self.s4_streams and self.n_nn_factors >= 2

    @property
    def n_shared_encoding_groups(self) -> int:
        """How many *distinct* NN input scopes are co-resident.

        Sharing an encoding is trivial when two NN factors are evaluated over
        the same coordinate grid and only differ in which columns they read.
        """
        return len({tuple(f.labels) for f in self.nn_factors})


@dataclass
class StreamingSurvey:
    problem: str
    strategy: str
    iB: int
    ecl: float
    max_merge_bound: Optional[int]
    n_vars: int
    clusters: List[ClusterRecord]

    # ---- aggregates -------------------------------------------------------
    @property
    def nn_clusters(self):
        return [c for c in self.clusters if c.path == 'nn']

    @property
    def exact_clusters(self):
        return [c for c in self.clusters if c.path == 'exact']

    @property
    def n_nn(self):
        return len(self.nn_clusters)

    @property
    def s1_stream_clusters(self):
        return [c for c in self.clusters if c.s1_streams]

    @property
    def s4_stream_clusters(self):
        return [c for c in self.clusters if c.s4_streams]

    @property
    def s1_joint_clusters(self):
        return [c for c in self.clusters if c.s1_joint]

    @property
    def s4_joint_clusters(self):
        return [c for c in self.clusters if c.s4_joint]

    def max_cross_block_redundancy(self, site='s1') -> float:
        vals = [getattr(f, f'{site}_redundancy')
                for c in self.clusters
                for f in c.nn_factors
                if (c.s1_streams if site == 's1' else c.s4_streams)]
        return max(vals) if vals else 1.0

    def summary(self) -> Dict:
        return {
            'problem': self.problem,
            'strategy': self.strategy,
            'iB': self.iB,
            'ecl': self.ecl,
            'max_merge_bound': self.max_merge_bound,
            'n_vars': self.n_vars,
            'n_clusters': len(self.clusters),
            'n_nn_clusters': self.n_nn,
            'max_co_resident_nn': max((c.n_nn_factors for c in self.clusters), default=0),
            'n_clusters_ge2_nn': sum(1 for c in self.clusters if c.n_nn_factors >= 2),
            'n_s1_stream': len(self.s1_stream_clusters),
            'n_s4_stream': len(self.s4_stream_clusters),
            'n_s1_joint': len(self.s1_joint_clusters),
            'n_s4_joint': len(self.s4_joint_clusters),
            'max_s1_redundancy': self.max_cross_block_redundancy('s1'),
            'max_s4_redundancy': self.max_cross_block_redundancy('s4'),
        }


# ---------------------------------------------------------------------------
# the survey itself
# ---------------------------------------------------------------------------
def _dispatch_path(gm, bucket) -> str:
    """The real ``FastGM.process_bucket`` gate (graphical_model.py ~line 436)."""
    if (bucket.get_width() <= gm.iB and bucket.get_ec() <= gm.ecl) \
            or bucket.get_message_complexity() < gm.complexity_limit:
        return 'exact'
    return gm.config.get('approximation_method', 'nn')


def survey_gm(gm, problem: str = '', strategy: str = '',
              max_merge_bound: Optional[int] = None) -> StreamingSurvey:
    """Replay the elimination bookkeeping on ``gm`` and record cluster structure.

    CONSUMES ``gm`` (empties ``gm.buckets``), exactly as a real elimination does.
    No message is computed, no network trained, no tensor allocated.
    """
    dom = {v.label: int(v.states) for v in gm.vars}
    records: List[ClusterRecord] = []

    for var in list(gm.elim_order):
        if var not in gm.buckets:
            continue                      # absorbed by a merge, or already done
        b = gm.buckets[var]

        # --- structure AT DISPATCH TIME (all earlier messages have arrived) --
        elim_labels = [getattr(v, 'label', v) for v in b.elim_vars]
        # built exactly as compute_message_exact builds it
        scope = sorted(set().union(*[set(f.labels) for f in b.factors])) if b.factors else []
        message_scope = b.get_message_scope()
        path = _dispatch_path(gm, b)

        elim_prod = _prod(elim_labels, dom)
        joint_numel = _prod(scope, dom)

        nn_facs = [f for f in b.factors if getattr(f, 'is_nn', False)]

        rec = ClusterRecord(
            key_label=getattr(var, 'label', var),
            elim_labels=sorted(elim_labels),
            message_scope=list(message_scope),
            scope=scope,
            n_factors=len(b.factors),
            n_nn_factors=len(nn_facs),
            path=path,
            width=len(message_scope),
            message_size=float(b.get_ec()),
            elim_prod=elim_prod,
            joint_numel=joint_numel,
        )

        # ---- S1: sample-gen streaming, only on the NN branch ---------------
        if path == 'nn' and elim_prod > SAMPLEGEN_DENSE_LIMIT:
            rec.s1_streams = True
            rec.s1_n_blocks = elim_prod

        # ---- S4: chunked exact message, only on the exact branch -----------
        batch_set = set()
        if path == 'exact' and joint_numel > EXACT_JOINT_NUMEL_LIMIT:
            bl, be, nb = plan_exact_blocks(scope, elim_labels, dom)
            rec.s4_streams = True
            rec.s4_batch_labels, rec.s4_batch_elim, rec.s4_n_blocks = bl, be, nb
            batch_set = set(bl) | set(be)

        # ---- per-NN-factor cross-block reevaluation accounting -------------
        elim_set = set(elim_labels)
        for f in nn_facs:
            fr = NNFactorRecord(labels=list(f.labels), width=len(f.labels))
            present = [l for l in f.labels if l in elim_set]
            fr.s1_elim_vars_in_scope = len(present)
            fr.s1_distinct_points = _prod(present, dom)
            fr.s1_redundancy = (elim_prod / fr.s1_distinct_points) if rec.s1_streams else 1.0
            if rec.s4_streams:
                fixed = [l for l in f.labels if l in batch_set]
                free = [l for l in f.labels if l not in batch_set]
                fr.s4_free_labels = free
                fr.s4_rows_per_eval = _prod(free, dom)
                fr.s4_distinct_blocks = _prod(fixed, dom)
                fr.s4_redundancy = rec.s4_n_blocks / fr.s4_distinct_blocks
            rec.nn_factors.append(fr)

        records.append(rec)

        # --- deliver the (stub) message and retire the bucket, as the real
        #     eliminate_variables(all=True) does -----------------------------
        if message_scope:
            nxt = gm.find_next_bucket(list(message_scope), var)
            if nxt is not None:
                nxt.receive_message(_StubFactor(message_scope, path != 'exact', dom))
            # else: goes to root, which is never dispatched
        del gm.buckets[var]

    return StreamingSurvey(
        problem=problem, strategy=strategy, iB=gm.iB, ecl=float(gm.ecl),
        max_merge_bound=max_merge_bound, n_vars=len(gm.vars), clusters=records,
    )


def build_gm(problem_key: str, iB: int, ecl: float, strategy: str = 'nomerge',
             max_merge_bound: Optional[int] = None, extra_config: Optional[Dict] = None,
             device: str = 'cpu', catalog=None):
    """Build a fresh, un-eliminated FastGM for a catalogue problem.

    ``device='cpu'`` is the default because this is a *build-only* pass -- no
    message is ever computed, so the device is irrelevant to the result and CPU
    avoids occupying a GPU.  Pass ``device='cuda'`` if you intend to follow the
    survey with real computation.
    """
    from nce.benchmark_problems.catalog_utils import get_catalog
    from nce.inference.graphical_model import FastGM
    from nce.config_schema import prepare_config

    if strategy not in MERGE_STRATEGIES:
        raise ValueError(f'unknown strategy {strategy!r}; expected one of '
                         f'{sorted(MERGE_STRATEGIES)}')
    cfg = {
        'iB': int(iB), 'ecl': ecl, 'device': device,
        'approximation_method': 'nn', 'neurobe_mode': True,
        'num_samples': 'nbe,0.1', 'sampling_scheme': 'uniform',
        'stream_nn_exact': True, 'dope_factors': False, 'masked_net': True,
        'verbose_merge': False,
    }
    cfg.update(MERGE_STRATEGIES[strategy])
    if max_merge_bound is not None and strategy != 'nomerge':
        cfg['max_merge_bound'] = int(max_merge_bound)
    if extra_config:
        cfg.update(extra_config)

    catalog = catalog if catalog is not None else get_catalog()
    return FastGM(model=catalog[problem_key], nn_config=prepare_config(cfg, strict=False),
                  device=device)


def survey_problem(problem_key: str, iB: int, ecl: float, strategy: str = 'nomerge',
                   max_merge_bound: Optional[int] = None,
                   extra_config: Optional[Dict] = None, catalog=None) -> StreamingSurvey:
    """Build + survey in one call.  Seconds per problem, no GPU, no training."""
    gm = build_gm(problem_key, iB, ecl, strategy, max_merge_bound,
                  extra_config=extra_config, catalog=catalog)
    return survey_gm(gm, problem=problem_key, strategy=strategy,
                     max_merge_bound=max_merge_bound)
