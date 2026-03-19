---
phase: quick-26
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - /home/cohenn1/.nvm/versions/node/v24.9.0/lib/node_modules/gsd-pi/packages/native/dist/text/index.js
  - /home/cohenn1/NCE/scripts/patch-gsd-text-fallback.sh
autonomous: true
requirements: [FIX-GSD-TEXT-FALLBACK]

must_haves:
  truths:
    - "gsd --print executes without 'Native function wrapTextWithAnsi is not available' crash on linux-x64"
    - "All 6 text functions (wrapTextWithAnsi, truncateToWidth, sliceWithWidth, extractSegments, sanitizeText, visibleWidth) have JS fallbacks"
    - "A reusable patch script exists so the fix survives gsd-pi upgrades"
  artifacts:
    - path: "/home/cohenn1/.nvm/versions/node/v24.9.0/lib/node_modules/gsd-pi/packages/native/dist/text/index.js"
      provides: "JS fallback implementations for all 6 text functions"
      contains: "hasNativeText"
    - path: "/home/cohenn1/NCE/scripts/patch-gsd-text-fallback.sh"
      provides: "Reapplyable patch script for post-upgrade"
      min_lines: 10
  key_links:
    - from: "text/index.js"
      to: "../native.js"
      via: "hasNativeText flag check"
      pattern: "typeof native\\.wrapTextWithAnsi"
---

<objective>
Fix the GSD CLI crash "Native function 'wrapTextWithAnsi' is not available on linux-x64" by adding pure JavaScript fallback implementations to the text module.

Purpose: The native Rust addon requires GLIBC 2.33 but this system runs Ubuntu 20.04 (GLIBC 2.31). The json-parse and xxhash modules already have JS fallbacks for this case, but the text module does not. Without this fix, `gsd` commands crash on every invocation.

Output: Patched text/index.js with working JS fallbacks, plus a shell script to reapply the patch after gsd-pi upgrades.
</objective>

<execution_context>
@/home/cohenn1/.claude/get-shit-done/workflows/execute-plan.md
@/home/cohenn1/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@/home/cohenn1/.nvm/versions/node/v24.9.0/lib/node_modules/gsd-pi/packages/native/dist/text/index.js
@/home/cohenn1/.nvm/versions/node/v24.9.0/lib/node_modules/gsd-pi/packages/native/dist/json-parse/index.js
@/home/cohenn1/.nvm/versions/node/v24.9.0/lib/node_modules/gsd-pi/packages/native/dist/native.js
@/home/cohenn1/.nvm/versions/node/v24.9.0/lib/node_modules/gsd-pi/packages/native/dist/text/types.js

<interfaces>
<!-- The native.js module exports: -->
<!-- native: Proxy object (throws on function calls when addon fails to load) -->
<!-- nativeAvailable: boolean (BUT has a bug: set to true before try/catch, so unreliable) -->

<!-- The fallback detection pattern used by json-parse and xxhash: -->
<!-- const hasNativeX = typeof native.functionName === "function"; -->
<!-- This works because the Proxy returns a function that throws, so typeof === "function" -->
<!-- WAIT - the Proxy get() returns (..._args) => { throw ... } which IS a function -->
<!-- So typeof native.wrapTextWithAnsi === "function" would be TRUE even with the Proxy -->

<!-- CORRECTION: Need to use nativeAvailable from native.js OR catch the error -->
<!-- Actually looking more carefully at native.js: _loadedSuccessfully is set to true BEFORE -->
<!-- the require() call that might throw, so nativeAvailable is unreliable. -->
<!-- The Proxy's get returns an arrow function, so typeof check returns "function" for ALL props. -->

<!-- CORRECT APPROACH: Try calling the function in a try/catch at module load to detect: -->
<!-- try { native.visibleWidth("", 3); hasNativeText = true; } catch { hasNativeText = false; } -->

<!-- EllipsisKind enum from types.js: -->
<!-- EllipsisKind.Unicode = 0 (single char ellipsis) -->
<!-- EllipsisKind.Ascii = 1 ("..." ellipsis) -->
<!-- EllipsisKind.None = 2 (hard truncate) -->
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Patch text/index.js with JS fallback implementations</name>
  <files>/home/cohenn1/.nvm/versions/node/v24.9.0/lib/node_modules/gsd-pi/packages/native/dist/text/index.js</files>
  <action>
Replace the entire text/index.js with a version that includes JS fallback implementations for all 6 functions. Follow the pattern from json-parse/index.js and xxhash/index.js.

CRITICAL DETECTION ISSUE: The Proxy in native.js returns arrow functions for ALL property accesses, so `typeof native.wrapTextWithAnsi === "function"` will always be true even when native is unavailable. Instead, detect native availability by attempting an actual call in a try/catch at module load time:

```javascript
let hasNativeText = false;
try { native.visibleWidth("test", 3); hasNativeText = true; } catch { hasNativeText = false; }
```

Implement these 6 pure-JS fallback functions:

1. **visibleWidth(text, tabWidth=3)** - Calculate visible width ignoring ANSI escape sequences. Strip ANSI codes using regex `/\x1b\[[0-9;]*[a-zA-Z]/g`, then count characters. Handle tab characters as tabWidth spaces. Handle East Asian wide characters if feasible (width 2), but basic ASCII-width counting is acceptable as a fallback.

2. **sanitizeText(text)** - Strip ANSI escapes, remove control characters (0x00-0x1F except \t \n), remove lone surrogates, normalize line endings (remove \r). Return original string if no changes needed.

3. **wrapTextWithAnsi(text, width, tabWidth=3)** - Word-wrap text respecting visible width. For the JS fallback: split into lines, for each line check visible width against `width`. If over, break at word boundaries (spaces). ANSI codes should pass through (not count toward width). Return the wrapped string with newlines. A simplified implementation is acceptable -- exact ANSI carry-across-lines is nice-to-have but not required for the fallback to be functional.

4. **truncateToWidth(text, maxWidth, ellipsisKind=0, pad=false, tabWidth=3)** - Truncate to maxWidth visible columns. If truncated and ellipsisKind is 0, append unicode ellipsis. If 1, append "...". If 2, hard truncate. If pad is true, pad with spaces to exactly maxWidth. Walk character by character, tracking visible width (skip ANSI sequences).

5. **sliceWithWidth(line, startCol, length, strict=false, tabWidth=3)** - Extract a range of visible columns. Walk the string tracking visible column position. Collect characters where visible column is in [startCol, startCol+length). Pass through ANSI sequences encountered within the range.

6. **extractSegments(line, beforeEnd, afterStart, afterLen, strictAfter=false, tabWidth=3)** - Return an object with `before` (columns 0..beforeEnd) and `after` (columns afterStart..afterStart+afterLen) segments. Implement using sliceWithWidth internally: `before = sliceWithWidth(line, 0, beforeEnd)`, `after = sliceWithWidth(line, afterStart, afterLen, strictAfter, tabWidth)`. Return `{ before, after }`.

Each exported function should check `hasNativeText` and delegate to native when available, falling back to the JS implementation otherwise. Structure:

```javascript
export function visibleWidth(text, tabWidth) {
    if (hasNativeText) return native.visibleWidth(text, tabWidth);
    return jsVisibleWidth(text, tabWidth);
}
```

ANSI regex for stripping: `/\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07|\x1b[()][AB012]/g` (covers SGR, OSC, charset sequences).

Keep the existing `export { EllipsisKind } from "./types.js";` line.
  </action>
  <verify>
Run `timeout 90 gsd --print 2>&1 | head -5` -- should produce output without "Native function" error. Also run `node -e "import('/home/cohenn1/.nvm/versions/node/v24.9.0/lib/node_modules/gsd-pi/packages/native/dist/text/index.js').then(m => { console.log('visibleWidth:', m.visibleWidth('hello', 3)); console.log('sanitize:', m.sanitizeText('test\x1b[31mred\x1b[0m')); console.log('wrap:', typeof m.wrapTextWithAnsi('hello world', 5, 3)); })"` -- all three should print without error.
  </verify>
  <done>All 6 text functions work via JS fallback. gsd CLI no longer crashes with "Native function not available" error. visibleWidth returns correct width for plain text, sanitizeText strips ANSI codes, wrapTextWithAnsi returns a string.</done>
</task>

<task type="auto">
  <name>Task 2: Create reusable patch script for gsd-pi upgrades</name>
  <files>/home/cohenn1/NCE/scripts/patch-gsd-text-fallback.sh</files>
  <action>
Create a bash script at `scripts/patch-gsd-text-fallback.sh` that:

1. Locates the gsd-pi text/index.js file (using the node path from `which gsd` or the known NVM path)
2. Backs up the original file as `text/index.js.orig` (only if backup doesn't already exist)
3. Copies the patched version over the original
4. Verifies the patch by running a quick node test

The script should:
- Be idempotent (safe to run multiple times)
- Auto-detect the gsd-pi install path: `$(dirname $(readlink -f $(which gsd)))/../lib/node_modules/gsd-pi/packages/native/dist/text/index.js` with fallback to the known NVM path
- Print clear success/failure messages
- Exit with code 0 on success, 1 on failure
- Include a comment at the top explaining WHY this patch exists (GLIBC 2.31 vs 2.33)

The script should embed the patched text/index.js content inline (heredoc) rather than copying from another file, so it's fully self-contained.

Mark executable with chmod +x.
  </action>
  <verify>Run `bash /home/cohenn1/NCE/scripts/patch-gsd-text-fallback.sh` -- should print success message. Run `file /home/cohenn1/NCE/scripts/patch-gsd-text-fallback.sh` to confirm it exists.</verify>
  <done>Patch script exists at scripts/patch-gsd-text-fallback.sh, is executable, and can reapply the text fallback patch after any gsd-pi upgrade. Script is self-contained with embedded patched file content.</done>
</task>

</tasks>

<verification>
1. `timeout 90 gsd --print 2>&1 | head -5` produces output (no crash)
2. `node -e "import('/home/cohenn1/.nvm/versions/node/v24.9.0/lib/node_modules/gsd-pi/packages/native/dist/text/index.js').then(m => console.log(m.visibleWidth('hello world', 3)))"` prints `11`
3. `bash /home/cohenn1/NCE/scripts/patch-gsd-text-fallback.sh` exits with code 0
</verification>

<success_criteria>
- gsd CLI commands execute without "Native function 'wrapTextWithAnsi' is not available on linux-x64" error
- All 6 text functions have working JS fallbacks
- Patch script exists and is rerunnable after future gsd-pi upgrades
</success_criteria>

<output>
After completion, create `.planning/quick/26-fix-gsd-cli-native-function-crash-on-lin/26-SUMMARY.md`
</output>
