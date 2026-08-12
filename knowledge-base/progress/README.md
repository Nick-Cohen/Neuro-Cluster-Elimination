# progress/ — synthesis layer

Turns the **daily** `lab_notebook.txt` into **weekly** summaries and **advisor-ready**
updates, so a week of work becomes an email or meeting update in minutes, and the paper's
"story of discovery" stays traceable.

## Workflow

1. **Daily:** keep logging raw work in `../../lab_notebook.txt` (unchanged habit).
2. **End of week:** copy `../templates/_weekly.md` → `weekly/<YYYY>-W<NN>.md` and synthesize
   that week's lab entries into it (headline, shipped, experiments, findings, blockers, next).
3. **Before a meeting / when asked:** copy `../templates/_advisor-update.md` →
   `advisor-updates/<YYYY-MM-DD>.md`, distilling the latest weekly note for a non-day-to-day
   audience (lead with results; define jargon).

## Conventions
- Weekly file = one ISO week. Get the week with `date -d <YYYY-MM-DD> +%G-W%V`.
- Every claimed number cites its lab-notebook date (provenance), matching `writeup/NOTES.md`.
- Link concepts with `[[wikilinks]]`; link the advisor update back to its weekly note.

## Index
- Weekly: `weekly/`  ·  Advisor updates: `advisor-updates/`
