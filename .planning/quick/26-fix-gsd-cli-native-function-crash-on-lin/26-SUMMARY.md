---
phase: quick-26
plan: 01
subsystem: tooling/gsd-cli
tags: [gsd, native-addon, glibc, js-fallback, patch]
dependency_graph:
  requires: []
  provides: [FIX-GSD-TEXT-FALLBACK]
  affects: [gsd-cli-usability]
tech_stack:
  added: []
  patterns: [try-catch-native-detection, embedded-heredoc-patch-script]
key_files:
  created:
    - /home/cohenn1/NCE/scripts/patch-gsd-text-fallback.sh
  modified:
    - /home/cohenn1/.nvm/versions/node/v24.9.0/lib/node_modules/gsd-pi/packages/native/dist/text/index.js
decisions:
  - "Use try/catch detection instead of typeof check: the Proxy in native.js returns arrow functions for ALL property accesses, so typeof native.wrapTextWithAnsi is always 'function' even when native is unavailable. Must attempt an actual call to detect availability."
  - "Embed patched content in script heredoc: makes patch-gsd-text-fallback.sh fully self-contained and runnable after upgrades without needing a separate source file."
metrics:
  duration: 8min
  completed: 2026-03-19
  tasks_completed: 2
  files_modified: 2
---

# Phase quick-26 Plan 01: Fix GSD CLI Native Function Crash Summary

**One-liner:** Pure-JS fallbacks for all 6 gsd-pi text functions using try/catch native detection, fixing GLIBC 2.31 incompatibility on Ubuntu 20.04.

## What Was Built

The gsd-pi text module only delegated to the native Rust addon with no JS fallbacks. On Ubuntu 20.04 (GLIBC 2.31), the addon requires GLIBC >= 2.33, so all text functions threw `Native function 'wrapTextWithAnsi' is not available on linux-x64` on every gsd invocation.

**Fix:** Added pure-JS implementations for all 6 text functions to the text/index.js module, with a reusable patch script to re-apply after gsd-pi upgrades.

## Tasks Completed

### Task 1: Patch text/index.js with JS fallback implementations

**File:** `/home/cohenn1/.nvm/versions/node/v24.9.0/lib/node_modules/gsd-pi/packages/native/dist/text/index.js`

Added 6 JS fallback functions:
- `jsVisibleWidth`: Strips ANSI via regex, counts character width including East Asian wide chars (width 2), handles tabs as tabWidth spaces
- `jsSanitizeText`: Strips ANSI regex, removes CR, removes control chars (except TAB/LF), removes lone surrogates
- `jsWrapTextWithAnsi`: Word-wraps at word boundaries, handles over-wide words by char-breaking, simplified (no ANSI state carry-across-lines)
- `jsTruncateToWidth`: Walks chars counting visible width, appends ellipsis (Unicode/ASCII/None) based on EllipsisKind, optional padding
- `jsSliceWithWidth`: Extracts visible column range [startCol, startCol+length), passes through ANSI sequences, handles wide-char boundary overlap
- `jsExtractSegments`: Returns {before, after} by calling jsSliceWithWidth twice

**Detection pattern:** Used try/catch at module load (not typeof) because the Proxy in native.js returns arrow functions for ALL property accesses, making typeof always return "function".

### Task 2: Create reusable patch script

**File:** `/home/cohenn1/NCE/scripts/patch-gsd-text-fallback.sh`

Self-contained bash script that:
- Auto-detects gsd-pi install path via `which gsd` symlink resolution, with NVM fallback
- Backs up original as `text/index.js.orig` (only if backup doesn't exist)
- Idempotent: skips re-patching if `hasNativeText` already present
- Embeds patched content as a heredoc (no external file dependency)
- Runs node verification after patching (visibleWidth, sanitizeText, wrapTextWithAnsi)
- Exits 0 on success, 1 on failure

## Verification Results

All plan verification checks passed:

1. `timeout 90 gsd --print` exits with code 0, no "Native function" crash
2. `node -e "import(...).then(m => console.log(m.visibleWidth('hello world', 3)))"` prints `11`
3. `bash scripts/patch-gsd-text-fallback.sh` exits with code 0

## Commits

| Hash | Description |
|------|-------------|
| 356a802 | feat(quick-26): add JS fallback patch for gsd-pi text module on GLIBC 2.31 |

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED

- `/home/cohenn1/NCE/scripts/patch-gsd-text-fallback.sh` exists and is executable
- `/home/cohenn1/.nvm/versions/node/v24.9.0/lib/node_modules/gsd-pi/packages/native/dist/text/index.js` contains `hasNativeText`
- Commit 356a802 exists in git log
