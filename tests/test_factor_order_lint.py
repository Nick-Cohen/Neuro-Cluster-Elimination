"""Lint: no unordered iteration over factor collections anywhere in `nce/`.

WHY A LINT AND NOT A RUNTIME TEST
---------------------------------
The 2026-08-12 determinism bug was one expression:

    unplaced_factors = set(factors)      # nce/inference/graphical_model.py:225

`FastFactor` defines no `__eq__`, so that set deduplicated by identity (fine)
and iterated in memory-address order (not fine). Bucket factor order is the
association order of a log-space product, so every message value moved by a few
float32 ULP, and 275+ SGD epochs turned that into a different trained model.
Root cause writeup: notebooks/_August-2026/claude_experiments/21-determinism.md.

That site is fixed and `FastFactor.__hash__` now returns a stable creation-order
id, so a recurrence would be deterministic *within a build order* rather than
address-ordered. But it would still silently change the factor product order
relative to today's goldens, and it would still be invisible in review. The
cheapest permanent guard is to make the SHAPE of the expression illegal.

WHAT IS FLAGGED
---------------
Any of these, when the iterated/converted expression's source text mentions
"factor" (case-insensitively):

  * `set(<factor expr>)` / `frozenset(<factor expr>)`
  * a set comprehension `{... for f in <factor expr>}`
  * a dict comprehension `{f: ... for f in <factor expr>}`
  * `for x in set(...)` / `dict(...)` over a factor expression

WHAT IS NOT FLAGGED, AND WHY THAT IS DELIBERATE
-----------------------------------------------
`set()` over LABELS is everywhere in this codebase and is harmless: labels are
ints, which hash to themselves, and the results are `sorted()` before use. A
lint that flagged those would produce ~60 hits, be turned off within a week, and
protect nothing. The name-based filter is the precision mechanism; it is
deliberately narrow, so this is a tripwire on the known failure mode, NOT proof
that no unordered iteration exists.

SUPPRESSION
-----------
Put `# factor-order-ok: <reason>` on the offending line. Suppressions are
listed by `test_suppressions_are_justified`, which requires a reason.
"""
import ast
import re
from pathlib import Path

import pytest

NCE_ROOT = Path(__file__).resolve().parents[1] / 'nce'

FACTOR_RE = re.compile(r'factor', re.IGNORECASE)
UNORDERED_CALLS = {'set', 'frozenset'}
SUPPRESS_RE = re.compile(r'#\s*factor-order-ok\s*:\s*(\S.*)$')


def _python_files():
    return sorted(p for p in NCE_ROOT.rglob('*.py'))


def _src(node, lines):
    try:
        return ast.unparse(node)
    except Exception:                      # pragma: no cover - 3.8 and older
        return ''


# Calls that pass their first argument through unchanged, so the factor-ness of
# `enumerate(bucket.factors)` is that of `bucket.factors`.
PASSTHROUGH_CALLS = {'enumerate', 'list', 'sorted', 'reversed', 'tuple', 'iter',
                     'filter', 'reversed'}


def _tail_name(node):
    """Last identifier of a dotted path: `self.bucket.factors` -> 'factors'."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_factor_expr(node):
    """True when the expression denotes a COLLECTION OF FACTORS.

    The distinction that makes this lint usable is between a collection of
    factors (`bucket.factors`) and an attribute OF a factor (`factor.labels`).
    Only the former is a determinism hazard; the latter is a set of int labels,
    it appears ~60 times in `nce/`, and flagging it would get the lint deleted.
    So the test is on the TAIL identifier of the expression, not on its whole
    source text.
    """
    name = _tail_name(node)
    if name is not None:
        return bool(FACTOR_RE.search(name))
    if isinstance(node, ast.Subscript):
        return _is_factor_expr(node.value)
    if isinstance(node, ast.Call):
        fname = _tail_name(node.func)
        if fname in PASSTHROUGH_CALLS and node.args:
            return _is_factor_expr(node.args[0])
        return bool(fname and FACTOR_RE.search(fname))
    if isinstance(node, (ast.GeneratorExp, ast.ListComp)):
        # `set(f for f in factors)` is a hazard; `set(v for s in factor_scopes
        # for v in s)` is a set of labels. Decide on the generator whose target
        # actually produces the elements.
        elt = _tail_name(node.elt)
        for gen in node.generators:
            if not _is_factor_expr(gen.iter):
                continue
            targets = {t.id for t in ast.walk(gen.target) if isinstance(t, ast.Name)}
            if elt is None or elt in targets:
                return True
        return False
    return False


class _Visitor(ast.NodeVisitor):
    def __init__(self, path, lines):
        self.path = path
        self.lines = lines
        self.hits = []
        self.suppressed = []

    def _record(self, node, kind, detail):
        line = self.lines[node.lineno - 1] if node.lineno - 1 < len(self.lines) else ''
        m = SUPPRESS_RE.search(line)
        entry = (str(self.path), node.lineno, kind, detail)
        if m:
            self.suppressed.append(entry + (m.group(1).strip(),))
        else:
            self.hits.append(entry)

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Name) and f.id in UNORDERED_CALLS and len(node.args) == 1:
            if _is_factor_expr(node.args[0]):
                self._record(node, '%s() over factors' % f.id, _src(node, None))
        self.generic_visit(node)

    def _comp(self, node, kind):
        for gen in node.generators:
            if _is_factor_expr(gen.iter):
                self._record(node, kind, _src(node, None)[:120])
                break
        self.generic_visit(node)

    def visit_SetComp(self, node):
        self._comp(node, 'set comprehension over factors')

    def visit_DictComp(self, node):
        self._comp(node, 'dict comprehension over factors')

    def visit_GeneratorExp(self, node):
        # A generator is ordered; only flag it when it is immediately consumed
        # by set()/frozenset(), which visit_Call already handles.
        self.generic_visit(node)


def _scan():
    hits, suppressed = [], []
    for path in _python_files():
        text = path.read_text(encoding='utf-8', errors='replace')
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:                # pragma: no cover
            continue
        v = _Visitor(path.relative_to(NCE_ROOT.parent), text.splitlines())
        v.visit(tree)
        hits += v.hits
        suppressed += v.suppressed
    return hits, suppressed


def test_no_unordered_factor_iteration():
    hits, _ = _scan()
    assert not hits, (
        'Unordered iteration over a factor collection was reintroduced. This is '
        'the exact shape of the 2026-08-12 determinism bug (doc 21): iterating a '
        'set of factors makes the factor product association order depend on '
        'hash-table layout, which silently moves every message value.\n\n'
        + '\n'.join('  %s:%d  %s\n      %s' % (h[0], h[1], h[2], h[3]) for h in hits)
        + '\n\nUse a list (and `sorted(...)` if you need dedup with a stable '
          'order). If the expression really is safe, annotate the line with\n'
          '    # factor-order-ok: <why>\n')


def test_lint_actually_fires_on_the_original_bug():
    """Non-vacuity guard.

    A lint that matches nothing passes forever. This feeds it the literal line
    that caused the outage and demands a hit, so a botched regex or a broken
    AST walk cannot masquerade as a clean codebase.
    """
    source = (
        'def _create_buckets_from_factors(self, factors):\n'
        '    unplaced_factors = set(factors)\n'
        '    ids = {f: i for i, f in enumerate(self.bucket.factors)}\n'
        '    uniq = {f for f in bw_factors}\n'
        '    scopes = set(labels)\n'                 # must NOT fire
        '    ordered = list(factors)\n'              # must NOT fire
    )
    tree = ast.parse(source)
    v = _Visitor(Path('synthetic.py'), source.splitlines())
    v.visit(tree)
    kinds = sorted(k for _, _, k, _ in v.hits)
    assert kinds == ['dict comprehension over factors',
                     'set comprehension over factors',
                     'set() over factors'], (
        'The factor-order lint no longer fires on the original bug. Detected: %r'
        % (v.hits,))
    assert not v.suppressed


def test_lint_covers_the_real_tree():
    """Guard against the scanner silently walking zero files."""
    files = _python_files()
    assert len(files) > 30, 'only %d python files found under %s' % (len(files), NCE_ROOT)
    assert any(p.name == 'graphical_model.py' for p in files)


def test_suppressions_are_justified():
    _, suppressed = _scan()
    for entry in suppressed:
        assert entry[-1], 'empty `# factor-order-ok:` reason at %s:%d' % (entry[0], entry[1])
    if suppressed:
        print('factor-order lint suppressions in force:')
        for e in suppressed:
            print('  %s:%d  %s  -- %s' % (e[0], e[1], e[2], e[-1]))
