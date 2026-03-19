---
phase: quick
plan: 25
subsystem: tooling
tags: [gsd, cli, debugging, npm, jiti, startup-performance]
dependency_graph:
  requires: []
  provides: [gsd-cli-working]
  affects: []
tech_stack:
  added: []
  patterns:
    - "Use timeout 90 for gsd --print commands on Ubuntu 20.04"
    - "Use claude --print as a drop-in replacement when startup speed matters"
key_files:
  created: []
  modified: []
decisions:
  - "gsd is not hanging; it takes 47-52 seconds to start due to jiti TypeScript compilation"
  - "Use timeout 90 (not 30) for gsd --print on this system"
  - "claude --print is the recommended alternative when fast startup is needed"
metrics:
  duration: "35 min"
  completed: "2026-03-19"
  tasks_completed: 2
  files_modified: 0
---

# Quick Task 25: Fix gsd CLI Not Working From Terminal — Summary

**One-liner:** gsd v2.32.0 is functional but requires 47-52 seconds to start on Ubuntu 20.04 due to jiti TypeScript transpilation of 438 extension files; use `timeout 90` or `claude --print` as alternatives.

## Tasks Completed

| # | Task | Status | Key Finding |
|---|------|--------|-------------|
| 1 | Update gsd-pi to v2.32.0 | Done | Updated from 2.31.2; still slow startup |
| 2 | Diagnose and document workaround | Done | Startup slow (47-52s), not hung; workarounds documented |

## Root Cause Analysis

### What Was Believed (Plan Context)
- `gsd --print` hangs indefinitely after printing the native addon warning
- The JS fallback for `parseStreamingJson` throws on call (proxy object)
- The streaming response handler silently fails

### What Actually Happens
- `gsd --print` does NOT hang — it completes successfully after ~47-52 seconds
- The apparent "hang" is a slow startup caused by jiti TypeScript compilation

### Native Addon Issue (GLIBC)
The system GLIBC is 2.31 (Ubuntu 20.04) but `@gsd-build/engine-linux-x64-gnu` requires GLIBC 2.33. When native addon fails to load, a JavaScript `Proxy` object is returned that throws for any function call. However:

- `parseStreamingJson` is only called for `tool_use` blocks (`input_json_delta` events) — NOT for plain text responses
- For `--print` mode with simple prompts, the native JSON parser is never invoked
- Therefore the Proxy fallback issue does NOT cause hangs for simple prompts

### Actual Root Cause: jiti Startup Cost

`gsd` loads 438 TypeScript extension files at startup using `@mariozechner/jiti` v2.6.5:

```
Location: ~/.gsd/agent/extensions/ (synced from dist/resources/extensions/)
Extensions: 20 directories (async-jobs, aws-auth, bg-shell, browser-tools, context7,
            get-secrets-from-user, google-search, gsd, mac-tools, mcp-client,
            mcporter, remote-questions, search-the-web, shared, slash-commands,
            subagent, ttsr, universal-config, voice)
Total files: 438 TypeScript files
```

jiti compiles each TypeScript file to an ES module and caches it at `/tmp/jiti/` (227 cache files). Even with the disk cache populated, Node.js must:
1. Stat each cached file
2. Read the compiled `.mjs` from disk
3. Parse and compile each module
4. Execute module initialization code

On this system (Ubuntu 20.04, kernel 5.4.0), this takes **47-52 seconds** due to the combination of:
- High file count (438 source → 227 compiled modules)
- jiti's loader overhead
- Node.js ESM module graph initialization

The `loader.js` has `moduleCache: false` hardcoded, which means even within the same process, compiled modules aren't cached in memory. However, changing this to `moduleCache: true` was tested and provided no improvement for CLI usage (each invocation is a fresh Node.js process, so in-memory caching doesn't persist).

## Timing Evidence

```
$ time gsd --no-session --print "say OK" 2>&1
[gsd] Native addon not available for linux-x64. Falling back to JS implementations (slower).
OK

real    0m47.125s
user    0m41.290s
sys     0m3.350s
```

The process runs at ~100-118% CPU (user space work, not blocked on I/O), state R (running), confirming it's computing, not waiting.

## Versions

```
gsd --version: 2.32.0 (updated from 2.31.2)
Node.js: v24.9.0
GLIBC: 2.31 (Ubuntu 20.04 LTS, kernel 5.4.0-150-generic)
Required GLIBC for native addon: 2.33
```

## Workarounds

### Option 1: Use Extended Timeout (Recommended)
```bash
# gsd works, just needs 90 seconds instead of 30
timeout 90 gsd --no-session --print "your prompt here" 2>&1
```

### Option 2: Use claude --print (Faster Alternative)
```bash
# claude --print starts instantly (~1-2 seconds)
claude --print "your prompt here"
```

`claude --print` is the official Anthropic CLI (Claude Code), which is already installed and working. It uses the same Anthropic API and model selection. For scripting and automation where startup speed matters, this is the preferred alternative.

### Option 3: Use gsd headless (For gsd-specific slash commands)
```bash
# For gsd-specific slash commands that require gsd's extension system
timeout 120 gsd headless /gsd:your-command
```

## What Does NOT Work
- `timeout 30 gsd --print ...` — exits with code 124 (timeout), no output
- Disabling extensions: gsd cli.js does not expose `--no-extensions` flag
- Patching `moduleCache: false` to `true`: no improvement for CLI usage

## Deviations from Plan

None — the plan accurately described the investigation steps. The conclusion ("document the alternative command") was reached after systematic diagnosis.

## Self-Check: PASSED

- PLAN.md: Present at `.planning/quick/25-fix-gsd-command-not-working-from-termina/25-PLAN.md`
- Task 1 complete: `gsd --version` returns 2.32.0
- Task 2 complete: Root cause documented, workarounds documented
- `gsd --no-session --print "say OK"` verified to produce "OK" output (within 90s)
- `claude --print "say OK"` verified to work as fast alternative
