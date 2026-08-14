#!/usr/bin/env python
"""Doc 50: emit driver specs for the memorization-threshold sweep.

spec = problem|merge|D|seed|arm|samplecap|memcap|samplefrac|memfrac|tag|outdir

Design:
  * mem_frac = K/2^w, the ENTRY fraction Nick capped at 10%.
  * sample_frac = 2 * mem_frac  -> constant 2:1 oversampling, so the ONLY thing
    that varies across the sweep is K, and the table-build cost grows with K
    (the honest cost of a bigger table).
  * one extra "anchor" arm at mem_frac=0.01 with sample_frac=0.1, which is doc
    44's exact configuration (10:1) -- both a comparability anchor and a
    fixed-K probe of the pool-size effect.
  * no absolute caps: every NN cluster on both cells has 2^w <= 2^20, so
    K_max = 104858 and N_max = 209716.

usage: python 50_specs.py <cell> [seeds...]
"""
import sys

CELLS = {
    'g20': ('grids/grid20x20.f10', 'sub', 10, 'results50_g20'),
    'g10w': ('grids/grid10x10.f10.wrap', 'sub', 10, 'results50_g10w'),
}
POINTS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1]


def tag(mf):
    return 'k%05d' % round(mf * 100000)


def specs(cell, seeds, with_base=True, with_anchor=True, points=POINTS):
    prob, merge, D, odir = CELLS[cell]
    out = []
    for s in seeds:
        if with_base:
            out.append(f'{prob}|{merge}|{D}|{s}|base|0|0|0.1|0.01|base|{odir}')
        for mf in points:
            out.append(f'{prob}|{merge}|{D}|{s}|mem|0|0|{2 * mf:g}|{mf:g}|'
                       f'{tag(mf)}|{odir}')
        if with_anchor:
            out.append(f'{prob}|{merge}|{D}|{s}|mem|0|0|0.1|0.01|anchor|{odir}')
    return out


if __name__ == '__main__':
    cell = sys.argv[1]
    seeds = [int(x) for x in sys.argv[2:]] or [42, 43, 44]
    for s in specs(cell, seeds):
        print(s)
