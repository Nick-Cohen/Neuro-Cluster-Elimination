---
phase: quick
plan: 25
type: execute
wave: 1
depends_on: []
files_modified: []
autonomous: true
requirements: [QUICK-25]
must_haves:
  truths:
    - "gsd CLI command runs without hanging in print mode"
    - "gsd CLI command can execute slash commands via headless mode"
    - "Native addon warning is resolved or confirmed non-blocking"
  artifacts: []
  key_links: []
---

<objective>
Fix the `gsd` CLI tool (gsd-pi npm package) which hangs when invoked from the terminal.

Purpose: The `gsd` command (v2.31.2, installed globally via npm as `gsd-pi`) hangs indefinitely
when run in `--print` or `headless` mode. The native Rust addon fails to load because the system
GLIBC version (2.31, Ubuntu 20.04) is older than what the addon requires (GLIBC 2.33). While the
tool claims to fall back to JS implementations, the fallback appears broken — the command hangs
after printing the native addon warning and never produces output or exits.

Output: Working `gsd` CLI command that can execute prompts from the terminal.
</objective>

<execution_context>
@/home/cohenn1/.claude/get-shit-done/workflows/execute-plan.md
@/home/cohenn1/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md

Diagnostic findings from planning:
- `gsd` v2.31.2 installed at `/home/cohenn1/.nvm/versions/node/v24.9.0/bin/gsd` (symlink to gsd-pi dist/loader.js)
- Native addon at `@gsd-build/engine-linux-x64-gnu` IS installed but requires GLIBC 2.33
- System has GLIBC 2.31 (Ubuntu 20.04, kernel 5.4.0-150-generic)
- `gsd --version` works (2.31.2)
- `gsd --list-models` works (lists Anthropic models, auth is valid)
- `gsd --print "hello"` hangs indefinitely (only prints native addon warning, then nothing)
- `gsd headless /gsd:help` also hangs
- Plain `gsd` (interactive) fails with "Interactive mode requires a terminal (TTY)" when no TTY
- OAuth token in `~/.gsd/agent/auth.json` is valid (expires 2026-03-19T07:18)
- Anthropic API is reachable (curl test returns expected responses)
- Latest available version: 2.32.0 (current: 2.31.2)
- The JS fallback for `parseStreamingJson` throws on call (proxy object), which likely causes the streaming response handler to silently fail
</context>

<tasks>

<task type="auto">
  <name>Task 1: Update gsd-pi to latest version and test</name>
  <files></files>
  <action>
1. Update `gsd-pi` to the latest version (2.32.0):
   ```bash
   npm install -g gsd-pi@latest
   ```

2. Verify the version:
   ```bash
   gsd --version
   ```
   Expected: 2.32.0

3. Test the native addon loading:
   ```bash
   gsd --list-models 2>&1 | head -5
   ```
   Check if the native addon warning is still present. The GLIBC issue (2.31 vs 2.33 required)
   may persist unless the new version ships a statically-linked addon or a musl build.

4. Test print mode (the main failing mode):
   ```bash
   timeout 30 gsd --print "say hello in exactly 3 words" 2>&1
   ```
   If this produces output and exits, the issue is fixed.

5. Test headless mode:
   ```bash
   timeout 30 gsd headless /gsd:help 2>&1
   ```

6. If the update fixes the issue, done. If not, proceed to Task 2.
  </action>
  <verify>
    Run `timeout 30 gsd --print "respond with OK" 2>&1` and confirm it produces a response
    (not just the native addon warning) and exits within 30 seconds.
  </verify>
  <done>
    `gsd --print` produces LLM output and exits cleanly, OR the update is confirmed as
    insufficient and Task 2 is needed.
  </done>
</task>

<task type="auto">
  <name>Task 2: Diagnose and work around the JS fallback hang (if update doesn't fix)</name>
  <files></files>
  <action>
If Task 1's update did not resolve the hang, investigate further:

1. Check if the new version still has the GLIBC issue:
   ```bash
   node -e "const {createRequire} = require('module'); const r = createRequire(require('path').join(require('os').homedir(), '.nvm/versions/node/v24.9.0/lib/node_modules/gsd-pi/packages/native/dist/native.js')); try { r('@gsd-build/engine-linux-x64-gnu'); console.log('NATIVE OK'); } catch(e) { console.log('NATIVE FAIL:', e.message.split('\n')[0]); }"
   ```

2. If GLIBC is still the issue, check if there's a musl or statically-linked variant:
   ```bash
   npm view @gsd-build/engine-linux-x64-gnu versions --json 2>&1 | tail -5
   npm ls @gsd-build/engine-linux-x64-gnu -g 2>&1
   ```

3. Try running with NODE_OPTIONS to enable verbose debugging:
   ```bash
   timeout 30 NODE_DEBUG=http,https,net gsd --print "hello" 2>&1 | tail -30
   ```
   This may reveal where the hang occurs (DNS resolution, TLS handshake, API response parsing).

4. If the hang is in the streaming JSON parser (JS fallback for native `parseStreamingJson`),
   check if there's an environment variable or config to disable streaming:
   ```bash
   grep -r "parseStreamingJson\|streaming.*json\|STREAM" ~/.nvm/versions/node/v24.9.0/lib/node_modules/gsd-pi/dist/ 2>/dev/null | head -20
   ```

5. Workarounds to try:
   a. Use `--mode text` instead of `--print` (different code path)
   b. Set `GSD_NO_NATIVE=1` or similar env var if supported
   c. Manually install a compatible native addon version
   d. Use `claude --print` (Claude Code CLI) as an alternative that has the same functionality

6. Document the working solution or file a bug report with gsd-pi maintainers.
  </action>
  <verify>
    Run `timeout 30 gsd --print "respond with OK" 2>&1` and confirm it produces output,
    OR document the alternative command that works (e.g., `claude --print`).
  </verify>
  <done>
    Either `gsd` CLI works from the terminal, or a documented workaround exists and the user
    knows how to use it.
  </done>
</task>

</tasks>

<verification>
- `gsd --version` returns a version number
- `gsd --print "say OK"` produces LLM output within 30 seconds
- If using a workaround, the alternative command is documented
</verification>

<success_criteria>
The `gsd` CLI command can be invoked from the terminal to run prompts and/or slash commands
without hanging indefinitely.
</success_criteria>

<output>
After completion, create `.planning/quick/25-fix-gsd-command-not-working-from-termina/25-SUMMARY.md`
</output>
