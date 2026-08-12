# NCE Knowledge Base (Obsidian vault)

This folder is the project's **Zettelkasten / slip-box** for NCE — neural-network
approximation of inference in discrete probabilistic graphical models. It is a plain
folder of wikilinked Markdown; open it directly as an [Obsidian](https://obsidian.md)
vault (point Obsidian at `knowledge-base/`).

## What this vault is for

1. **Writing & citation support** — correct terminology, canonical citations, and
   precise definitions so any author (human or Claude) can write papers, related-work
   sections, and rebuttals without re-deriving the field. Citations live in
   [`references.bib`](references.bib); every source has a matching literature note.
2. **Progress that is easy to summarize** — a weekly-synthesis and advisor-update
   layer (`progress/`) so a week of work can be turned into an email or meeting update
   in minutes, and so the paper's "story of discovery" is traceable.

It **complements** the rest of the repo (it does not replace it):
`lab_notebook.txt` = daily raw log · `docs/` = API/config reference ·
`.gsd/` + `.planning/` = project management · `writeup/` = the paper draft ·
**this vault = durable conceptual + bibliographic + synthesis knowledge.**

> This vault is the canonical successor to the old hidden `.knowledge/` graph; those
> notes were migrated here (see `.knowledge/README.md` pointer stub).

## Folder layout

| Folder | Note type | One note = |
|---|---|---|
| `00-maps/` | Map of Content (MOC) | a curated entry-point into a topic cluster |
| `literature/` | Literature note (`@key.md`) | one **source** (paper/book), summarized in our words + how it relates to NCE |
| `concepts/` | Permanent / atomic note | one **idea**, written in our own words, densely linked |
| `glossary/` | Glossary | term → canonical definition, correct usage, and how *we* use it |
| `project/` | Project note | NCE-specific knowledge (our method, terminology map, codebase map, open questions) |
| `patterns/` | Implementation pattern | a recurring code/design convention in `nce/` |
| `progress/` | Synthesis | `weekly/` summaries, `advisor-updates/`, from the lab notebook |
| `templates/` | Note templates | copy when creating a new note |

## Conventions (slip-box rules)

- **Atomic:** one idea per concept note. If a note needs an "and", consider splitting.
- **Own words:** concept notes are written, not quoted. Quotes go in literature notes
  with attribution.
- **Link liberally:** connect with `[[wikilinks]]`. A link to a not-yet-written note is
  fine — it marks a future note. The graph view shows what's well-connected vs orphaned.
- **Cite explicitly:** any non-obvious claim links to a literature note `[[@key]]`,
  whose `citekey` matches `references.bib`. Concept notes end with a `## Sources` section.
- **Separate "the field" from "our project":** a concept note describes the published
  idea; a `## In NCE` section (or a link to a `project/` note) says how the codebase
  uses it. Where our terminology differs from the literature, flag it and link
  [[terminology-map]].
- **Frontmatter** (YAML) on every note — see templates. `status` is one of
  `seedling` (stub), `growing` (usable, incomplete), `evergreen` (stable).

## Entry points

- Start at [[MOC-home]].
- New to the area? [[MOC-exact-inference]] → [[MOC-approximate-inference]] →
  [[MOC-neural-inference]] → [[MOC-this-project]].
- Need a citation? [`references.bib`](references.bib) or browse `literature/`.
- Need to write an update? `progress/weekly/` and `templates/_weekly.md`.
