"""Reusable structural-analysis machinery for NCE bucket/cluster structure.

Everything in here is *build-only*: it inspects the post-merge cluster
structure and simulates the elimination bookkeeping WITHOUT computing any
message, training any network, or allocating any message tensor. A full
survey of a benchmark problem takes well under a second.

Public entry points:

    from nce.analysis import survey_problem, survey_gm, StreamingSurvey

See ``nce/analysis/structure_survey.py`` for details.
"""
from .structure_survey import (          # noqa: F401
    ClusterRecord,
    NNFactorRecord,
    StreamingSurvey,
    build_gm,
    survey_gm,
    survey_problem,
    plan_exact_blocks,
    MERGE_STRATEGIES,
)
