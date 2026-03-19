#!/usr/bin/env bash
# patch-gsd-text-fallback.sh
#
# PURPOSE:
#   Applies JS fallback implementations to the gsd-pi text/index.js module.
#
# WHY THIS EXISTS:
#   The gsd-pi native Rust addon requires GLIBC >= 2.33, but Ubuntu 20.04
#   ships with GLIBC 2.31. When the addon fails to load, native.js returns a
#   Proxy that throws on every function call instead of providing JS fallbacks.
#   The json-parse and xxhash modules already have JS fallbacks, but the text
#   module does not. This patch adds pure-JS implementations for all 6 text
#   functions (visibleWidth, sanitizeText, wrapTextWithAnsi, truncateToWidth,
#   sliceWithWidth, extractSegments) using the correct try/catch detection
#   pattern (typeof detection doesn't work because the Proxy always returns
#   an arrow function for every property).
#
# USAGE:
#   bash scripts/patch-gsd-text-fallback.sh
#
#   Safe to run multiple times (idempotent). Run after any gsd-pi upgrade.
#
# DETECTING PATCH STATUS:
#   The patched file contains the string "hasNativeText". The original does not.

set -euo pipefail

# ── Locate gsd-pi text/index.js ───────────────────────────────────────────────

# Try to auto-detect via the gsd binary location
GSD_BIN=$(which gsd 2>/dev/null || true)
TEXT_INDEX=""

if [ -n "$GSD_BIN" ]; then
    # Follow symlinks to find the real binary, then navigate to gsd-pi package
    GSD_REAL=$(readlink -f "$GSD_BIN" 2>/dev/null || echo "$GSD_BIN")
    GSD_DIR=$(dirname "$GSD_REAL")
    # Try <gsd-bin-dir>/../lib/node_modules/gsd-pi/...
    CANDIDATE="$GSD_DIR/../lib/node_modules/gsd-pi/packages/native/dist/text/index.js"
    if [ -f "$CANDIDATE" ]; then
        TEXT_INDEX=$(realpath "$CANDIDATE")
    fi
fi

# Fallback: known NVM path for Node v24.9.0
if [ -z "$TEXT_INDEX" ] || [ ! -f "$TEXT_INDEX" ]; then
    NVM_CANDIDATE="$HOME/.nvm/versions/node/v24.9.0/lib/node_modules/gsd-pi/packages/native/dist/text/index.js"
    if [ -f "$NVM_CANDIDATE" ]; then
        TEXT_INDEX="$NVM_CANDIDATE"
    fi
fi

if [ -z "$TEXT_INDEX" ] || [ ! -f "$TEXT_INDEX" ]; then
    echo "ERROR: Could not locate gsd-pi text/index.js. Is gsd-pi installed?" >&2
    echo "  Tried: gsd binary path and $HOME/.nvm/versions/node/v24.9.0/..." >&2
    exit 1
fi

echo "Found: $TEXT_INDEX"

# ── Check if already patched ─────────────────────────────────────────────────

if grep -q "hasNativeText" "$TEXT_INDEX" 2>/dev/null; then
    echo "Already patched (hasNativeText found). Running verification..."
    # Skip to verification
else
    # ── Backup original (only if no backup exists) ────────────────────────────
    BACKUP="${TEXT_INDEX}.orig"
    if [ ! -f "$BACKUP" ]; then
        echo "Backing up original to: $BACKUP"
        cp "$TEXT_INDEX" "$BACKUP"
    else
        echo "Backup already exists: $BACKUP"
    fi

    echo "Writing patched text/index.js..."

    # Write the patched content (embedded inline for self-contained operation)
    cat > "$TEXT_INDEX" << 'PATCHED_EOF'
/**
 * ANSI-aware text measurement and slicing.
 *
 * High-performance UTF-16 native implementation with ASCII fast-paths,
 * single-pass ANSI scanning, and proper Unicode grapheme cluster support.
 *
 * JS fallback: added for systems where the native Rust addon cannot load
 * due to GLIBC version mismatch (e.g. Ubuntu 20.04 has GLIBC 2.31 but
 * the native addon requires GLIBC 2.33). The json-parse and xxhash modules
 * use typeof-based detection, but the Proxy in native.js returns arrow
 * functions for ALL property accesses, so typeof always returns "function".
 * We detect availability by attempting an actual call in a try/catch.
 */
import { native } from "../native.js";
export { EllipsisKind } from "./types.js";
// Detect native availability by attempting a real call.
// typeof native.wrapTextWithAnsi === "function" is always true (Proxy returns arrow fn),
// so we must probe with an actual invocation.
let hasNativeText = false;
try {
    native.visibleWidth("", 3);
    hasNativeText = true;
}
catch {
    hasNativeText = false;
}
// ANSI escape sequence regex:
//   \x1b\[[0-9;]*[a-zA-Z]  - CSI sequences (SGR color codes, cursor movement, etc.)
//   \x1b\][^\x07]*\x07     - OSC sequences (terminal title, hyperlinks, etc.)
//   \x1b[()][ -~]          - Charset designation sequences
const ANSI_REGEX = /\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07|\x1b[()][^\x00]/g;
// Regex to match a single ANSI sequence at a given position
const ANSI_SINGLE = /^(\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07|\x1b[()][^\x00])/;
// East Asian wide character ranges (simplified)
function isWideChar(cp) {
    return ((cp >= 0x1100 && cp <= 0x115F) || // Hangul Jamo
        (cp >= 0x2E80 && cp <= 0x303E) || // CJK Radicals / Kangxi
        (cp >= 0x3040 && cp <= 0x33FF) || // Japanese / CJK Compatibility
        (cp >= 0x3400 && cp <= 0x4DBF) || // CJK Extension A
        (cp >= 0x4E00 && cp <= 0x9FFF) || // CJK Unified Ideographs
        (cp >= 0xA000 && cp <= 0xA4CF) || // Yi Syllables
        (cp >= 0xA960 && cp <= 0xA97F) || // Hangul Jamo Extended-A
        (cp >= 0xAC00 && cp <= 0xD7FF) || // Hangul Syllables + Jamo Ext-B
        (cp >= 0xF900 && cp <= 0xFAFF) || // CJK Compatibility Ideographs
        (cp >= 0xFE10 && cp <= 0xFE1F) || // Vertical Forms
        (cp >= 0xFE30 && cp <= 0xFE6F) || // CJK Compatibility Forms
        (cp >= 0xFF00 && cp <= 0xFF60) || // Fullwidth Forms
        (cp >= 0xFFE0 && cp <= 0xFFE6) || // Fullwidth Signs
        (cp >= 0x1B000 && cp <= 0x1B0FF) || // Kana Supplement
        (cp >= 0x1F004 && cp <= 0x1F0CF) || // Playing Cards (wide)
        (cp >= 0x1F300 && cp <= 0x1F9FF) || // Misc Symbols / Emoji
        (cp >= 0x20000 && cp <= 0x2A6DF) || // CJK Extension B
        (cp >= 0x2A700 && cp <= 0x2CEAF) || // CJK Extensions C/D/E
        (cp >= 0x2CEB0 && cp <= 0x2EBEF) || // CJK Extension F
        (cp >= 0x2F800 && cp <= 0x2FA1F) || // CJK Compatibility Supplement
        (cp >= 0x30000 && cp <= 0x3134F)); // CJK Extension G
}
/**
 * Calculate visible width of a single character (codepoint).
 */
function charWidth(cp, tabWidth) {
    if (cp === 9 /* TAB */)
        return tabWidth;
    if (cp < 32 || (cp >= 0x7F && cp <= 0x9F))
        return 0; // control chars
    if (cp === 0xAD)
        return 1; // soft hyphen
    if (isWideChar(cp))
        return 2;
    return 1;
}
/**
 * JS fallback: Calculate visible width of text excluding ANSI escape sequences.
 */
function jsVisibleWidth(text, tabWidth = 3) {
    if (typeof text !== "string")
        return 0;
    let width = 0;
    let i = 0;
    while (i < text.length) {
        // Check for ANSI escape sequence
        if (text[i] === "\x1b") {
            const rest = text.slice(i);
            const m = rest.match(ANSI_SINGLE);
            if (m) {
                i += m[1].length;
                continue;
            }
        }
        // Get codepoint (handle surrogate pairs)
        const cp = text.codePointAt(i) ?? 0;
        width += charWidth(cp, tabWidth);
        i += cp > 0xFFFF ? 2 : 1;
    }
    return width;
}
/**
 * JS fallback: Strip ANSI escapes, remove control characters, lone surrogates,
 * and normalize line endings (remove CR).
 */
function jsSanitizeText(text) {
    if (typeof text !== "string")
        return String(text);
    // Strip ANSI sequences
    let result = text.replace(ANSI_REGEX, "");
    // Remove CR
    result = result.replace(/\r/g, "");
    // Remove control characters except TAB (0x09) and LF (0x0A)
    result = result.replace(/[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]/g, "");
    // Remove lone surrogates (invalid Unicode)
    result = result.replace(/[\uD800-\uDFFF]/g, "");
    return result;
}
/**
 * JS fallback: Word-wrap text to a visible width, preserving ANSI escape codes.
 *
 * This is a simplified implementation. ANSI state carry-across-lines is best-effort.
 */
function jsWrapTextWithAnsi(text, width, tabWidth = 3) {
    if (typeof text !== "string")
        return String(text);
    if (width <= 0)
        return text;
    const inputLines = text.split("\n");
    const outputLines = [];
    for (const line of inputLines) {
        if (jsVisibleWidth(line, tabWidth) <= width) {
            outputLines.push(line);
            continue;
        }
        // Need to wrap this line
        const words = line.split(" ");
        let currentLine = "";
        let currentWidth = 0;
        for (let wi = 0; wi < words.length; wi++) {
            const word = words[wi];
            const wordWidth = jsVisibleWidth(word, tabWidth);
            const spaceWidth = currentLine.length > 0 ? 1 : 0;
            if (currentLine.length > 0 && currentWidth + spaceWidth + wordWidth > width) {
                outputLines.push(currentLine);
                currentLine = word;
                currentWidth = wordWidth;
            }
            else {
                if (currentLine.length > 0) {
                    currentLine += " ";
                    currentWidth += 1;
                }
                // If the word itself is wider than the limit, break it
                if (wordWidth > width) {
                    // Break word character by character
                    let i = 0;
                    while (i < word.length) {
                        const cp = word.codePointAt(i) ?? 0;
                        // Check for ANSI at this position
                        if (word[i] === "\x1b") {
                            const rest = word.slice(i);
                            const m = rest.match(ANSI_SINGLE);
                            if (m) {
                                currentLine += m[1];
                                i += m[1].length;
                                continue;
                            }
                        }
                        const cw = charWidth(cp, tabWidth);
                        if (currentWidth + cw > width && currentLine.length > 0) {
                            outputLines.push(currentLine);
                            currentLine = "";
                            currentWidth = 0;
                        }
                        const ch = cp > 0xFFFF ? word.slice(i, i + 2) : word[i];
                        currentLine += ch;
                        currentWidth += cw;
                        i += cp > 0xFFFF ? 2 : 1;
                    }
                }
                else {
                    currentLine += word;
                    currentWidth += wordWidth;
                }
            }
        }
        if (currentLine.length > 0) {
            outputLines.push(currentLine);
        }
    }
    return outputLines.join("\n");
}
/**
 * JS fallback: Truncate text to maxWidth visible columns.
 */
function jsTruncateToWidth(text, maxWidth, ellipsisKind = 0, pad = false, tabWidth = 3) {
    if (typeof text !== "string")
        text = String(text);
    const totalWidth = jsVisibleWidth(text, tabWidth);
    if (totalWidth <= maxWidth) {
        if (pad) {
            return text + " ".repeat(maxWidth - totalWidth);
        }
        return text;
    }
    // Determine ellipsis
    let ellipsis = "";
    let ellipsisWidth = 0;
    if (ellipsisKind === 0) {
        ellipsis = "\u2026";
        ellipsisWidth = 1;
    }
    else if (ellipsisKind === 1) {
        ellipsis = "...";
        ellipsisWidth = 3;
    }
    // else ellipsisKind === 2: no ellipsis
    const targetWidth = maxWidth - ellipsisWidth;
    if (targetWidth <= 0) {
        const result = ellipsis.slice(0, maxWidth);
        if (pad)
            return result + " ".repeat(maxWidth - jsVisibleWidth(result, tabWidth));
        return result;
    }
    // Walk characters collecting up to targetWidth visible columns
    let result = "";
    let currentWidth = 0;
    let i = 0;
    while (i < text.length) {
        // Check for ANSI escape sequence - pass through without counting width
        if (text[i] === "\x1b") {
            const rest = text.slice(i);
            const m = rest.match(ANSI_SINGLE);
            if (m) {
                result += m[1];
                i += m[1].length;
                continue;
            }
        }
        const cp = text.codePointAt(i) ?? 0;
        const cw = charWidth(cp, tabWidth);
        if (currentWidth + cw > targetWidth) {
            break;
        }
        const ch = cp > 0xFFFF ? text.slice(i, i + 2) : text[i];
        result += ch;
        currentWidth += cw;
        i += cp > 0xFFFF ? 2 : 1;
    }
    result += ellipsis;
    if (pad) {
        const resultWidth = currentWidth + ellipsisWidth;
        result += " ".repeat(maxWidth - resultWidth);
    }
    return result;
}
/**
 * JS fallback: Slice a range of visible columns from a line.
 */
function jsSliceWithWidth(line, startCol, length, strict = false, tabWidth = 3) {
    if (typeof line !== "string")
        return "";
    if (length <= 0)
        return "";
    const endCol = startCol + length;
    let result = "";
    let currentCol = 0;
    let i = 0;
    while (i < line.length) {
        // Check for ANSI escape sequence
        if (line[i] === "\x1b") {
            const rest = line.slice(i);
            const m = rest.match(ANSI_SINGLE);
            if (m) {
                // Pass through ANSI if we're in the desired range
                if (currentCol >= startCol && currentCol < endCol) {
                    result += m[1];
                }
                i += m[1].length;
                continue;
            }
        }
        const cp = line.codePointAt(i) ?? 0;
        const cw = charWidth(cp, tabWidth);
        const charEnd = currentCol + cw;
        // Check if this character falls within [startCol, endCol)
        if (currentCol >= startCol && charEnd <= endCol) {
            const ch = cp > 0xFFFF ? line.slice(i, i + 2) : line[i];
            result += ch;
        }
        else if (currentCol >= startCol && currentCol < endCol && charEnd > endCol) {
            // Wide character overlaps boundary
            if (!strict) {
                const ch = cp > 0xFFFF ? line.slice(i, i + 2) : line[i];
                result += ch;
            }
        }
        currentCol += cw;
        if (currentCol >= endCol)
            break;
        i += cp > 0xFFFF ? 2 : 1;
    }
    return result;
}
/**
 * JS fallback: Extract the before/after segments around an overlay region.
 */
function jsExtractSegments(line, beforeEnd, afterStart, afterLen, strictAfter = false, tabWidth = 3) {
    const before = jsSliceWithWidth(line, 0, beforeEnd, false, tabWidth);
    const after = jsSliceWithWidth(line, afterStart, afterLen, strictAfter, tabWidth);
    return { before, after };
}
/**
 * Word-wrap text to a visible width, preserving ANSI escape codes across
 * line breaks.
 *
 * Active SGR codes (colors, bold, etc.) are carried to continuation lines.
 * Underline and strikethrough are reset at line ends and restored on the
 * next line.
 */
export function wrapTextWithAnsi(text, width, tabWidth) {
    if (hasNativeText)
        return native.wrapTextWithAnsi(text, width, tabWidth);
    return jsWrapTextWithAnsi(text, width, tabWidth);
}
/**
 * Truncate text to a visible width with an optional ellipsis.
 *
 * @param text       Input string (may contain ANSI codes).
 * @param maxWidth   Maximum visible width in terminal cells.
 * @param ellipsisKind  0 = "\u2026", 1 = "...", 2 = none.
 * @param pad        When true, pad with spaces to exactly `maxWidth`.
 * @param tabWidth   Tab stop width (default 3, range 1-16).
 */
export function truncateToWidth(text, maxWidth, ellipsisKind, pad, tabWidth) {
    if (hasNativeText)
        return native.truncateToWidth(text, maxWidth, ellipsisKind, pad, tabWidth);
    return jsTruncateToWidth(text, maxWidth, ellipsisKind, pad, tabWidth);
}
/**
 * Slice a range of visible columns from a line.
 *
 * Counts terminal cells (skipping ANSI escapes). When `strict` is true,
 * wide characters that would exceed the range are excluded.
 */
export function sliceWithWidth(line, startCol, length, strict, tabWidth) {
    if (hasNativeText)
        return native.sliceWithWidth(line, startCol, length, strict, tabWidth);
    return jsSliceWithWidth(line, startCol, length, strict, tabWidth);
}
/**
 * Extract the before/after segments around an overlay region.
 *
 * ANSI state is tracked so the `after` segment renders correctly even when
 * the overlay truncates styled text.
 */
export function extractSegments(line, beforeEnd, afterStart, afterLen, strictAfter, tabWidth) {
    if (hasNativeText)
        return native.extractSegments(line, beforeEnd, afterStart, afterLen, strictAfter, tabWidth);
    return jsExtractSegments(line, beforeEnd, afterStart, afterLen, strictAfter, tabWidth);
}
/**
 * Strip ANSI escape sequences, remove control characters and lone
 * surrogates, and normalize line endings (CR removed).
 *
 * Returns the original string when no changes are needed (zero-copy).
 */
export function sanitizeText(text) {
    if (hasNativeText)
        return native.sanitizeText(text);
    return jsSanitizeText(text);
}
/**
 * Calculate visible width of text excluding ANSI escape sequences.
 *
 * Tabs count as `tabWidth` cells (default 3).
 */
export function visibleWidth(text, tabWidth) {
    if (hasNativeText)
        return native.visibleWidth(text, tabWidth);
    return jsVisibleWidth(text, tabWidth);
}
PATCHED_EOF

    echo "Patch written successfully."
fi

# ── Verify the patch ──────────────────────────────────────────────────────────

echo "Verifying patch..."

VERIFY_OUTPUT=$(node -e "
import('$TEXT_INDEX').then(m => {
    const w = m.visibleWidth('hello world', 3);
    if (w !== 11) { console.error('FAIL: visibleWidth expected 11, got ' + w); process.exit(1); }
    const s = m.sanitizeText('test\x1b[31mred\x1b[0m');
    if (s !== 'testred') { console.error('FAIL: sanitizeText expected testred, got ' + s); process.exit(1); }
    const wrap = m.wrapTextWithAnsi('hello world', 5, 3);
    if (typeof wrap !== 'string') { console.error('FAIL: wrapTextWithAnsi returned non-string'); process.exit(1); }
    console.log('OK: visibleWidth=11, sanitizeText=testred, wrapTextWithAnsi=string');
    process.exit(0);
}).catch(e => { console.error('FAIL:', e.message); process.exit(1); });
" 2>&1)

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Verification PASSED: $VERIFY_OUTPUT"
    echo ""
    echo "Patch applied successfully to: $TEXT_INDEX"
    echo "gsd CLI text functions will use JS fallbacks on this system."
    exit 0
else
    echo "Verification FAILED: $VERIFY_OUTPUT" >&2
    echo "The patch may not have been applied correctly." >&2
    exit 1
fi
