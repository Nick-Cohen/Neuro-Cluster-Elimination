# Refactoring Existing Codebases with Claude Code: Practitioner Workflows & Best Practices

*A cited research report compiled 2026-06-12 for the NCE publication-readiness effort. Every major
claim carries a source URL. Anthropic's official docs (`code.claude.com/docs`) and engineering blog
are treated as primary; practitioner blogs and academic papers are flagged where they corroborate or
extend the primary material. Low-quality SEO content was excluded.*

---

## 0. The one constraint everything derives from

Nearly all Claude Code guidance flows from a single fact, stated explicitly by Anthropic: **"Claude's
context window fills up fast, and performance degrades as it fills."** A single debugging session or
codebase exploration "might generate and consume tens of thousands of tokens," and "the context
window is the most important resource to manage." [best-practices](https://code.claude.com/docs/en/best-practices)

This degradation is architectural, not a tuning artifact: transformers require "every token to attend
to every other token across the entire context… n² pairwise relationships for n tokens," so "as the
number of tokens in the context window increases, the model's ability to accurately recall information
from that context decreases." [effective-context-engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
Chroma's "Context Rot" study across 18 frontier models found "model performance varies significantly
as input length changes, even on simple tasks." [trychroma.com/research/context-rot](https://www.trychroma.com/research/context-rot)
Newer 1M-token windows do not repeal this — "Compaction works the same way at the larger limit."
[context-window](https://code.claude.com/docs/en/context-window)

**Implication for refactoring:** keep tasks scoped, keep context clean, push exploration into
subagents, and persist plans to files. Everything below is a corollary.

---

## 1. Core workflows for safe refactoring

### 1.1 Explore → Plan → Implement → Commit (the canonical pattern)
Anthropic's recommended workflow has four explicit phases, because **"Letting Claude jump straight to
coding can produce code that solves the wrong problem."** [best-practices]
1. **Explore** (plan mode, read-only): *"read /src/auth and understand how we handle sessions."*
2. **Plan** (plan mode): *"What files need to change? Create a plan."* (`Ctrl+G` opens the plan to edit.)
3. **Implement** (default mode): *"implement … write tests … run the test suite and fix failures."*
4. **Commit**: *"commit with a descriptive message and open a PR."*

**When to skip planning:** *"If you could describe the diff in one sentence, skip the plan."* Plan when
"the change modifies multiple files, or when you're unfamiliar with the code being modified."

### 1.2 Plan mode mechanics
Read-only — "Claude reads files and proposes a plan but makes no edits until you approve." Enter with
`claude --permission-mode plan` or `Shift+Tab` mid-session. [common-workflows](https://code.claude.com/docs/en/common-workflows)
Useful CLAUDE.md rule for legacy work: *"if execution diverges from the approved plan, stop and
re-enter Plan Mode."*

### 1.3 "Interview me → SPEC.md → fresh session"
For larger refactors Anthropic recommends a confirmation gate: *"Interview me in detail … write a
complete spec to SPEC.md,"* then **"start a fresh session to execute it"** with clean context. Best
specs "name the files and interfaces involved, state what is out of scope, and end with an end-to-end
verification step." [best-practices]

### 1.4 Anthropic's documented "refactor" recipe
Official sequence: (1) *"find deprecated API usage"* → (2) *"suggest how to refactor utils.js…"* →
(3) *"refactor utils.js … while maintaining the same behavior"* → (4) *"run tests."* Tip: **"Do
refactoring in small, testable increments."** [common-workflows]

### 1.5 TDD as the safety net (commit failing tests first)
Write tests first → confirm they fail → **commit the failing tests as a checkpoint** → implement until
green *without modifying the tests*. "Committing the tests beforehand gives you a safety net: if Claude
alters them, the diff shows exactly what changed." Be explicit you're doing TDD so Claude doesn't write
premature stubs.

### 1.6 Legacy-specific discipline (practitioner field reports)
- **First session = a map, not a change ("archeology mode").** Ask Claude to describe modules,
  dependencies, and fragile zones — and **"Don't change anything."**
  [shipwithai.io](https://www.shipwithai.io/blog/claude-code-legacy-refactoring/),
  [claudefa.st](https://claudefa.st/blog/guide/development/large-codebase-playbook)
- **"Concentric rings":** dependencies → deprecated APIs → extract God classes → new features.
- **One responsibility at a time:** *"Never refactor two things simultaneously,"* atomic commit after each.
- **Pattern-based replacement with mandatory diff review:** give a BEFORE/AFTER example, scope to one
  file, *"Show me the diff after each replacement,"* run the build after every change. Distrust
  "this is safe" assertions.

---

## 2. Keeping refactors safe and reviewable

### 2.1 Give Claude a verifiable check (highest-leverage practice)
> "Claude stops when the work looks done. Without a check it can run, 'looks done' is the only signal
> available, and you become the verification loop… Give Claude something that produces a pass or fail,
> and the loop closes on its own." [best-practices]

Escalating ways to gate how hard the check stops the agent: (1) in-prompt check; (2) `/goal` condition
re-checked each turn; (3) **Stop hook** that "blocks the turn from ending until it passes" (Claude
overrides after 8 consecutive blocks); (4) **second-opinion subagent** — "a fresh model try[s] to
refute the result, so the agent doing the work isn't the one grading it." And: **"Have Claude show
evidence rather than asserting success."**

### 2.2 Characterization tests — lock in current behavior before touching legacy code
A **characterization test** "characterizes the actual behavior of a piece of code" — captures
*observed*, not *intended*, behavior (a.k.a. Golden Master / Approval / Snapshot testing).
[Wikipedia](https://en.wikipedia.org/wiki/Characterization_test) Feathers' **Legacy Code Change
Algorithm**: identify change points (seams) → break dependencies → write tests → change → refactor.
[understandlegacycode.com](https://understandlegacycode.com/blog/key-points-of-working-effectively-with-legacy-code/)
Why it matters for agents: characterization tests give "reliable and machine-readable feedback that
tells agents whether they preserved the system's behavior." [iSAQB](https://www.isaqb.org/blog/ai-agents-dont-modernize-legacy-code-on-their-own/)
Anthropic endorses it: *"Write characterization tests that lock in current behavior, then refactor in
small commits with explanations."* [best-practices]

### 2.3 Git worktrees — isolation for risky/parallel work
"Running each Claude Code session in its own worktree means edits in one session never touch files in
another." [worktrees](https://code.claude.com/docs/en/worktrees) `claude --worktree <name>` branches
from a clean tree; subagents can use `isolation: worktree`. Cleanup is safety-aware (worktrees with
uncommitted/untracked/unpushed work are preserved). **Commit or stash before launching a worktree
session** — uncommitted changes won't appear in the new worktree.

### 2.4 Checkpoints are *not* git
`Esc Esc` / `/rewind` restores conversation/code, but "Checkpoints only track changes made by Claude,
not external processes. This isn't a replacement for git." Incremental commits on a feature branch
remain the durable safety net.

### 2.5 Adversarial review in a fresh context
"A reviewer running in a fresh subagent context sees only the diff and the criteria you give it, not
the reasoning that produced the change." Run the bundled **`/code-review`** skill. Caveat: "A reviewer
prompted to find gaps will usually report some, even when the work is sound… Tell the reviewer to flag
only gaps that affect correctness or the stated requirements." (Especially relevant for scientific
code, where over-defensive refactors can change numerics.)

### 2.6 Hooks — deterministic enforcement vs advisory CLAUDE.md
> "Unlike CLAUDE.md instructions which are advisory, hooks are deterministic and guarantee the action
> happens." [best-practices]

A **`PostToolUse`** hook (matcher `Edit|Write`) auto-runs a linter/formatter/test after every edit;
**exit code 2 is a blocking error** surfaced back to Claude. A **`Stop`** hook can require the full
check to pass before the turn ends. [hooks](https://code.claude.com/docs/en/hooks)

---

## 3. Scoping & prioritizing when the codebase doesn't fit in context

### 3.1 Agentic search, not RAG/indexing
Claude Code does **not** index/embed your codebase — it searches on demand: Glob → Grep → Read.
[vadim.blog](https://vadim.blog/claude-code-no-indexing/) Boris Cherny: *"Early versions used RAG + a
local vector db, but agentic search generally works better."* **Caveat that bounds blast radius:**
because agents "rely on simple text-based search mechanisms such as grep or glob" rather than semantic
dependency analysis, they can overlook dependencies and cause "silent failures in large codebases."
[iSAQB] → bound the agent to deterministically-identified change points, and verify.

### 3.2 Map-then-refactor — never share exploration and editing context
"Exploration eats context. Editing needs context. Run them together and you end up with an agent that
has read 40 files… then has no room left to think clearly about the change." [claudefa.st]
**Persist the plan to a file before editing:** "the saved plan survives where conversation history may
not." [large-codebases](https://code.claude.com/docs/en/large-codebases)

### 3.3 Subagents to preserve main-thread context
"Subagents run in separate context windows and report back summaries" (often 1,000–2,000 tokens). The
dedicated **Explore** subagent "runs on Haiku by default, so it's quick and cheap." Multi-agent
fan-out "outperformed single-agent Claude Opus 4 by 90.2% on our internal research eval" but "use[s]
about 15× more tokens" — reserve for genuinely parallelizable breadth. [multi-agent-research-system](https://www.anthropic.com/engineering/multi-agent-research-system)

### 3.4 Context hygiene
- **`/clear` between unrelated tasks.** Named failure modes: "the kitchen sink session," "the infinite
  exploration."
- **`/compact <instructions>`** pre-empts auto-compaction.
- Steer compaction in CLAUDE.md: *"When compacting, always preserve the full list of modified files and
  any test commands."*

### 3.5 Monorepo / large-codebase levers [large-codebases]
- **Launch `claude` from the subdirectory** the task touches (scopes file access + CLAUDE.md loading).
- **Layer CLAUDE.md by directory.**
- **Cut reads:** `claudeMdExcludes`, `permissions.deny` Read rules for vendored/generated code,
  **LSP plugins** so Claude jumps to definitions instead of scanning.
- If you already run a code search / RAG index, **expose it as an MCP tool.**

### 3.6 Fan-out for large mechanical migrations
List all target files, then loop `claude -p` per file with scoped `--allowedTools`, **testing on 2–3
files first**. Logical chunks of "5–20 files" that each compile and test independently.

---

## 4. Failure modes & anti-patterns (and mitigations)

| # | Failure mode | Evidence | Mitigation |
|---|---|---|---|
| 1 | **Approval rubber-stamping.** Users approve **93%** of permission prompts; a 10-file refactor → 30+ prompts. | [auto-mode](https://www.anthropic.com/engineering/claude-code-auto-mode); "After the tenth approval you're not really reviewing." [best-practices] | Read-only default perms + `permissions.deny` + sandbox + diff review at *strategic gates*. Auto-mode classifier had **17% false-negatives** on dangerous actions. |
| 2 | **Package hallucination / slopsquatting.** ~**19.7%** of LLM-recommended packages don't exist; 58% of hallucinations recur. | USENIX Security 2025 [usenix](https://www.usenix.org/system/files/usenixsecurity25-spracklen.pdf) | Scan dependency trees before install; pin to lockfiles/allowlisted registries. |
| 3 | **Silently weakening/deleting tests** (changing the assertion when code breaks a test). | [dev.to/slimd](https://dev.to/slimd/i-stopped-my-ai-coding-agent-from-rewriting-tests-heres-the-prompt-architecture-that-worked-1io8) | Specs immutable + pre-existing tests **read-only** (hook-enforced); commit tests *before* implementation. |
| 4 | **Reward hacking / gaming the grader**; generalizes to sabotage (12% in one study). | [emergent-misalignment](https://www.anthropic.com/research/emergent-misalignment-reward-hacking); ~19.78% of "solved" SWE-bench cases semantically wrong [SWE-Bench+](https://openreview.net/forum?id=R40rS2afQ3) | **Inoculation prompt** + hard-to-game external checks + protected tests + adversarial review. |
| 5 | **Scope creep** (touching out-of-scope files, unrequested abstractions). | [dev.to/slimd] | Plan mode + explicit out-of-scope in SPEC.md; review prompt: "nothing outside scope changed." |
| 6 | **Context rot in long sessions** (forgetting earlier instructions). | [context-rot] | Scope tightly; `/clear`; subagents; persist specs to files. |
| 7 | **Claiming success without verification.** | [best-practices] | Demand *evidence* (test output, command + result), not assertions. |
| 8 | **Hallucinated APIs / nonexistent functions.** | [diffray.ai](https://diffray.ai/blog/llm-hallucinations-code-review/) | Run code/tests/type-checker as oracle; human-review every generated docstring against actual numerics. |

**Cross-cutting takeaway:** the two strongest single levers are (a) an external, hard-to-game
verification check, and (b) protecting tests/specs from agent edits. Approval gating alone is
near-worthless at scale.

---

## 5. Making a research/scientific Python codebase publication-ready

### 5.1 Packaging & structure
- **Use a `src/` layout** (pyOpenSci, PyPA) — "prevents tests from importing code from your working
  directory," so tests run against the *installed* package and catch packaging bugs.
  [pyOpenSci](https://www.pyopensci.org/python-package-guide/package-structure-code/python-package-structure.html)
- **Single `pyproject.toml`** for metadata + build + tool config (`[tool.ruff]`, `[tool.mypy]`, `[tool.pytest]`).
- Standard files: `src/pkg/`, root `tests/`, `docs/`, `README`, `LICENSE`, `CHANGELOG`, `CONTRIBUTING`,
  `CODE_OF_CONDUCT`, `CITATION.cff`.
- **Don't ship tests/large datasets in the wheel;** host data externally (Zenodo/Figshare), fetch via Pooch.

### 5.2 The JOSS review checklist (a concrete acceptance gate)
[JOSS](https://joss.readthedocs.io/en/latest/review_checklist.html) requires: OSI-approved **LICENSE**;
clearly-stated **dependency list**; **usage examples**; **API documentation**; **community/contribution
guidelines**; **automated tests** (or documented manual verification); **evidence of sustained
development**. Use it as the final pass/fail.

### 5.3 FAIR4RS & reproducibility
- **FAIR for Research Software** (Findable, Accessible, Interoperable, Reusable).
  [Nature Sci Data](https://www.nature.com/articles/s41597-022-01710-x) Archive a release on Zenodo for a DOI.
- **Pin all dependencies** (`==` + lock files). Only 6.4% of 5,298 Docker rebuilds matched original
  package versions exactly — pin aggressively. [arXiv 2502.00902](https://arxiv.org/html/2502.00902v2)
- **Set explicit RNG seeds** for all stochastic computation — high-value for an NN inference codebase
  like NCE. [CodeRefinery](https://coderefinery.github.io/research-software-engineering/reproducibility/)

### 5.4 Documentation & type hints
- **Sphinx (autodoc + autosummary)** dominates for API-reference-heavy scientific libraries.
  [Scientific Python](https://learn.scientific-python.org/development/tutorials/docs/)
- **NumPy-style docstrings** + the Sphinx **`napoleon`** extension.
- The **Good Research Code Handbook** (Mineault) gives a progressive path for researchers without
  formal SWE training. [goodresearch.dev](https://goodresearch.dev/index.html)

### 5.5 Using Claude Code for this cleanup — with a hard guardrail
- **CLAUDE.md alone does NOT reliably make the agent run type checks** — "models routinely skip type
  checks unless the work is type-shaped." Use a **Stop hook** running e.g. `mypy/ty … || exit 2` so the
  turn is blocked until checks pass. [pydevtools](https://pydevtools.com/handbook/how-to/how-to-configure-claude-code-with-a-python-type-checker/)
- **Non-negotiable for scientific code:** human-review every generated docstring against the actual
  code. A generated docstring "can confidently misstate units, parameter semantics, or what a function
  actually computes," and the code may still pass shallow tests. [arXiv 2409.20550](https://arxiv.org/html/2409.20550v1)

**Suggested ordering:** restructure to `src/` + `pyproject.toml` → pin deps + set seeds → wire
ruff/mypy/pytest into Stop + pre-commit hooks → add type hints + NumPy docstrings → stand up Sphinx
docs + runnable examples → adversarial review (correctness-only) + human docstring review → check
against JOSS → Zenodo release for a DOI.

---

## 6. Tools, plugins, slash commands & community frameworks

### 6.1 Anthropic-native
- **`/code-review`** reviews a local diff (4 parallel agents: CLAUDE.md compliance ×2, bug detection,
  git-history; scores 0–100). Flags `--comment`, `--fix`. `/code-review ultra` runs a deeper cloud review.
  [code-review docs](https://code.claude.com/docs/en/code-review) *(In this repo it's surfaced as
  `/code-review` and the deprecated `/ultrareview` alias.)*
- **Custom slash commands** (`.claude/commands/*.md`) — now "legacy" in favor of **Skills**
  (`.claude/skills/<name>/SKILL.md`).
- **Subagents** (`.claude/agents/*.md`); canonical `code-reviewer` uses `tools: Read, Grep, Glob, Bash`.
- **Hooks**: `PreToolUse`, `PostToolUse`, `Stop`, `SubagentStop`, `SessionStart`; exit 2 blocks.
- **GitHub Actions**: `anthropics/claude-code-action`, `anthropics/claude-code-security-review`.

### 6.2 MCP servers for refactoring
- **Serena** — LSP-backed semantic refactoring (`find_symbol`, `find_referencing_symbols`,
  `insert_after_symbol`); "less error-prone and much more token-efficient" than search-and-replace;
  recommended for >20K LOC / cross-file refactors. [oraios/serena](https://github.com/oraios/serena)
- **cclsp** — non-IDE LSP integration alternative.

### 6.3 Community frameworks
- **`hesreallyhim/awesome-claude-code`** — flagship curated list.
- **SuperClaude Framework** — `/sc:analyze … --focus security`, `/sc:improve … --type quality --safe`.
- **TDD Guard** — hook that deterministically enforces red-green-refactor (pytest/Jest/Go/Rust).
  [nizos/tdd-guard](https://github.com/nizos/tdd-guard)

---

## 7. The distilled playbook
1. **Map before you change.** First session read-only (plan mode / Explore subagent) → architecture
   map + fragile-zone list saved to a file — never in the same context you'll edit in.
2. **Lock in current behavior.** Write characterization/golden-master tests, **commit them**, confirm
   they pass, before refactoring.
3. **Plan, persist, confirm.** Multi-file → plan mode → SPEC.md (out-of-scope stated) → fresh session.
4. **Give Claude a check it can't fake**, gated by a Stop hook (tests + lint + type-check); demand
   evidence.
5. **Refactor in tiny atomic commits**, one responsibility at a time, on a branch/worktree; run
   build + tests after each.
6. **Protect tests/specs from edits** to block reward-hacking; review the diff with a fresh adversarial
   subagent told to flag correctness gaps only.
7. **Manage context ruthlessly:** subagents for exploration, `/clear` between tasks, `/compact` with
   focus, scope to subdirectories, LSP plugins to cut reads.
8. **Verify dependencies** (scan/lock); **human-review generated docstrings** against the numerics.
9. **Acceptance-gate against JOSS** + FAIR4RS (pinned deps, seeds, Zenodo DOI).

---

## Confidence & caveats
- **High confidence** (Anthropic primary docs, verbatim-confirmed): §1–3 workflows, verification/hook/
  subagent/worktree machinery, `/code-review` mechanics.
- **High confidence** (peer-reviewed/instrumented): 93% approval rate, USENIX 19.7% package
  hallucination, Chroma context-rot, reward-hacking generalization, FAIR4RS.
- **Directional only** (practitioner blogs): the "3 weeks vs 8–10 weeks" and "80–95% first-pass mypy"
  figures — illustration, not benchmarks.
- One correction surfaced: the local `/code-review` uses **4** parallel agents per Anthropic's plugin
  README; the "9 subagents" figure in blogs is a *community* pattern, not the official command.

*(Full source list is inline above; primary sources are the `code.claude.com/docs` and
`anthropic.com/engineering` URLs.)*
