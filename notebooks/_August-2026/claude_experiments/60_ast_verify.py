#!/usr/bin/env python
"""
60_ast_verify.py -- structural verification of the frozen/rerun-v1 merges.

Doc 53's lesson: a merge can apply "cleanly" (or be hand-resolved plausibly) and
still produce a method with no body, or a duplicate definition that shadows the
real one. Reading the diff does not catch that. So: parse base / parent1 /
parent2 / merged, and compare FUNCTION BODIES by qualified name.

For every function in every file touched by more than one merged branch:
  - merged body must equal parent1's, or parent2's, or -- when both parents
    changed the same function -- must be exactly base + parent1's added lines
    + parent2's added lines (a genuine both-sides combine, verified line-wise).
  - no function may match NO source.
  - the function inventory must be exactly accounted for (no dropped defs, no
    duplicate defs).

Usage: python 60_ast_verify.py <base_sha> <parent1_sha> <parent2_sha>
Run from the merged worktree with the merge staged/committed.
"""
import ast
import subprocess
import sys
from collections import Counter

BASE, P1, P2 = sys.argv[1], sys.argv[2], sys.argv[3]

# Files touched by >= 2 of the branches merged into frozen/rerun-v1.
FILES = [
    "nce/inference/factor.py",
    "nce/inference/bucket.py",
    "nce/inference/graphical_model.py",
    "nce/inference/message_gradient_factors.py",
    "nce/utils/backward_message.py",
    "nce/sampling/sample_generator.py",
    "tests/test_wmb_merge_repair.py",
]


def show(rev, path):
    r = subprocess.run(["git", "show", f"{rev}:{path}"],
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def read_merged(path):
    with open(path) as fh:
        return fh.read()


def bodies(src):
    """qualname -> (normalized body via ast.unparse, raw source lines of body)."""
    if src is None:
        return None
    tree = ast.parse(src)
    lines = src.splitlines()
    out = {}
    dupes = Counter()

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = f"{prefix}{child.name}"
                dupes[q] += 1
                body = child.body
                # strip a leading docstring
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    body = body[1:]
                norm = "\n".join(ast.unparse(s) for s in body) if body else "<EMPTY>"
                raw = lines[child.lineno - 1: child.end_lineno]
                out[q] = (norm, raw)
                walk(child, q + ".")
            elif isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")

    walk(tree, "")
    return out, dupes


def added_lines(base_raw, other_raw):
    """Lines present in other_raw beyond base_raw (multiset difference)."""
    c = Counter(l.strip() for l in other_raw) - Counter(l.strip() for l in base_raw)
    return c


ok = True
combined = []
totals = Counter()

for path in FILES:
    sb, s1, s2 = show(BASE, path), show(P1, path), show(P2, path)
    sm = read_merged(path)
    rb = bodies(sb)
    r1 = bodies(s1)
    r2 = bodies(s2)
    rm = bodies(sm)
    bb = rb[0] if rb else {}
    b1 = r1[0] if r1 else {}
    b2 = r2[0] if r2 else {}
    bm, dupes_m = rm

    # --- duplicate definition check (doc 53's duplicate _get_slices failure) ---
    dupes = {k: v for k, v in dupes_m.items() if v > 1}
    if dupes:
        ok = False
        print(f"FAIL {path}: duplicate definitions {dupes}")

    # --- empty-body check (doc 53's bodyless _elim_table failure) ---
    empties = [k for k, (n, _) in bm.items() if n == "<EMPTY>"]
    legit_empty = [k for k in empties
                   if (k in b1 and b1[k][0] == "<EMPTY>") or (k in b2 and b2[k][0] == "<EMPTY>")]
    bad_empty = sorted(set(empties) - set(legit_empty))
    if bad_empty:
        ok = False
        print(f"FAIL {path}: functions with no body not present in any source: {bad_empty}")

    # --- inventory ---
    lost = (set(b1) | set(b2)) - set(bm)
    # a def deleted on purpose by one parent is only "lost" if the other kept it unchanged
    real_lost = [k for k in lost if (k in b1 and k in b2)]
    if real_lost:
        ok = False
        print(f"FAIL {path}: definitions present in both parents but missing from merge: {sorted(real_lost)}")
    invented = set(bm) - set(b1) - set(b2)
    if invented:
        ok = False
        print(f"FAIL {path}: definitions in merge present in NEITHER parent: {sorted(invented)}")

    # --- body provenance ---
    for q, (nm, rawm) in sorted(bm.items()):
        n1 = b1.get(q, (None,))[0]
        n2 = b2.get(q, (None,))[0]
        totals["checked"] += 1
        if nm == n1 or nm == n2:
            totals["matches_a_parent"] += 1
            continue
        # Neither parent matches -> must be a genuine both-sides combine.
        nb = bb.get(q, (None,))[0]
        if q not in bb or n1 is None or n2 is None:
            ok = False
            print(f"FAIL {path}::{q}: body matches no parent and is not a base-derived combine")
            continue
        rawb = bb[q][1]
        a1 = added_lines(rawb, b1[q][1])
        a2 = added_lines(rawb, b2[q][1])
        am = added_lines(rawb, rawm)
        expect = a1 + a2
        if am == expect:
            # also confirm nothing base-side was dropped
            dropped = (Counter(l.strip() for l in rawb)
                       - Counter(l.strip() for l in rawm))
            dropped_1 = Counter(l.strip() for l in rawb) - Counter(l.strip() for l in b1[q][1])
            dropped_2 = Counter(l.strip() for l in rawb) - Counter(l.strip() for l in b2[q][1])
            if dropped and dropped != (dropped_1 + dropped_2) and not (dropped <= (dropped_1 + dropped_2)):
                ok = False
                print(f"FAIL {path}::{q}: combine dropped base lines neither parent dropped: {list(dropped)[:5]}")
            else:
                totals["verified_combine"] += 1
                combined.append(f"{path}::{q}  (base + {sum(a1.values())} lines from P1 + {sum(a2.values())} from P2)")
        else:
            ok = False
            print(f"FAIL {path}::{q}: not a clean both-sides combine.")
            print(f"   expected added: {sorted(expect.elements())}")
            print(f"   actual   added: {sorted(am.elements())}")

print()
print("=" * 72)
print(f"files checked            : {len(FILES)}")
print(f"function bodies checked  : {totals['checked']}")
print(f"  match a parent exactly : {totals['matches_a_parent']}")
print(f"  verified both-sides    : {totals['verified_combine']}")
for c in combined:
    print(f"      {c}")
print("=" * 72)
print("RESULT:", "PASS -- zero merge damage" if ok else "FAIL")
sys.exit(0 if ok else 1)
