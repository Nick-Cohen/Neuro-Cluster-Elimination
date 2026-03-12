---
phase: quick-24
plan: 24
type: execute
wave: 1
depends_on: []
files_modified:
  - .planning/GSD_EXPORT.md
autonomous: true
requirements: []

must_haves:
  truths:
    - "A single self-contained markdown file captures the complete GSD planning state for migration to GSD 2.0"
    - "All project definitions, requirements, roadmap, state, phase plans, config, and quick task history are preserved"
    - "The export is human-readable and can be used to reconstruct planning context in a new GSD version"
  artifacts:
    - path: ".planning/GSD_EXPORT.md"
      provides: "Complete export of all GSD planning artifacts"
      min_lines: 200
  key_links: []
---

<objective>
Create a comprehensive markdown export of all GSD planning artifacts so the project can be resumed after upgrading to GSD 2.0.

Purpose: Preserve all planning state, decisions, requirements, roadmap, phase plans, and quick task history in a single self-contained file that can be used to bootstrap the project in a new GSD version.

Output: `.planning/GSD_EXPORT.md` containing all planning artifacts.
</objective>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/REQUIREMENTS.md
@.planning/STATE.md
@.planning/config.json
@.planning/phases/05-config-restructure/05-01-PLAN.md
@.planning/phases/05-config-restructure/05-02-PLAN.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Assemble comprehensive GSD export file</name>
  <files>.planning/GSD_EXPORT.md</files>
  <action>
Create `.planning/GSD_EXPORT.md` that assembles ALL planning artifacts into a single self-contained markdown file. The file structure should be:

```
# GSD Planning Export — NCE Project
## Export Metadata
- Export date, GSD version (1.0), project name, export purpose

## 1. Project Definition
- Full contents of .planning/PROJECT.md (verbatim)

## 2. Requirements
- Full contents of .planning/REQUIREMENTS.md (verbatim)

## 3. Roadmap
- Full contents of .planning/ROADMAP.md (verbatim)

## 4. Project State
- Full contents of .planning/STATE.md (verbatim)

## 5. Planning Config
- Full contents of .planning/config.json (verbatim, in code block)

## 6. Phase Plans (Active)
### Phase 5: Config Restructure
#### Plan 05-01
- Full contents of 05-01-PLAN.md (verbatim, in code block)
#### Plan 05-02
- Full contents of 05-02-PLAN.md (verbatim, in code block)

## 7. Quick Task History
- The complete quick tasks table from STATE.md (already captured in Section 4)
- Additionally, list all quick task directories with their PLAN and SUMMARY file existence status:
  Read each quick task directory name from `.planning/quick/` (excluding task 24 itself)
  For each, note: directory name, has PLAN (yes/no), has SUMMARY (yes/no)

## 8. Key Decisions Registry
- Extract and consolidate all decisions from STATE.md Accumulated Context section
- Group by category: config decisions, architecture decisions, algorithm decisions, experiment decisions

## 9. Migration Notes
- Note which phases are complete (1-4) and their plan files are no longer on disk
- Note phase 5 is planned but not started
- Note phases 6-9 have TBD plan counts
- Note v1.0 requirements are all complete, v1.1 requirements are all pending
- List any files/artifacts that were referenced but may need recreation in GSD 2.0
```

**Implementation approach:**
1. Read each source file using the Read tool
2. Embed contents verbatim in the appropriate sections
3. For the quick task manifest, list all directories in `.planning/quick/` and check for PLAN/SUMMARY files
4. For the decisions registry, parse STATE.md's decisions list and group them logically
5. Write the complete assembled file

**Important:** Embed file contents verbatim (inside markdown code fences where needed to avoid formatting conflicts). Do NOT summarize or abbreviate — the export must be complete enough to reconstruct the full planning state.
  </action>
  <verify>
    <automated>cd /home/cohenn1/NCE && python -c "
import os

# Check file exists
path = '.planning/GSD_EXPORT.md'
assert os.path.exists(path), f'{path} does not exist'

# Check file is substantial
content = open(path).read()
lines = content.split('\n')
assert len(lines) > 200, f'Export too short: {len(lines)} lines (expected >200)'

# Check all major sections present
required_sections = [
    'Project Definition',
    'Requirements',
    'Roadmap',
    'Project State',
    'Planning Config',
    'Phase Plans',
    'Quick Task History',
    'Key Decisions',
    'Migration Notes',
]
for section in required_sections:
    assert section in content, f'Missing section: {section}'

# Check key content is present (spot checks)
assert 'CFG-01' in content, 'Missing requirement CFG-01'
assert 'CFG2-01' in content, 'Missing requirement CFG2-01'
assert 'VIZ-01' in content, 'Missing requirement VIZ-01'
assert 'Phase 5: Config Restructure' in content, 'Missing Phase 5'
assert 'config_schema.py' in content, 'Missing config_schema.py reference'
assert 'flatten_config' in content, 'Missing flatten_config reference'
assert 'BenchmarkSet' in content, 'Missing BenchmarkSet reference'
assert 'Quick Tasks Completed' in content or 'quick' in content.lower(), 'Missing quick task history'

print(f'Export validated: {len(lines)} lines, all sections present')
print('All checks passed')
"
    </automated>
  </verify>
  <done>
    - .planning/GSD_EXPORT.md exists with 200+ lines
    - Contains all 9 sections: Project Definition, Requirements, Roadmap, State, Config, Phase Plans, Quick Task History, Key Decisions, Migration Notes
    - All source file contents are embedded verbatim
    - Quick task manifest lists all 23 completed tasks with PLAN/SUMMARY status
    - Decisions are grouped by category
    - Migration notes identify what needs attention in GSD 2.0
  </done>
</task>

</tasks>

<verification>
1. `.planning/GSD_EXPORT.md` exists and is >200 lines
2. All 9 major sections are present
3. Source file contents are embedded verbatim (not summarized)
4. Quick task manifest is complete
5. Key decisions are consolidated and grouped
6. Migration notes identify gaps and next steps
</verification>

<success_criteria>
- A single file at `.planning/GSD_EXPORT.md` contains the complete GSD planning state
- The file is self-contained — no external file reads needed to understand the project's planning state
- All requirements, roadmap phases, state decisions, and task history are preserved
- Migration notes clearly identify what's complete, what's pending, and what needs attention
</success_criteria>

<output>
After completion, create `.planning/quick/24-export-gsd-planning-artifacts-for-gsd-2-/24-SUMMARY.md`
</output>
