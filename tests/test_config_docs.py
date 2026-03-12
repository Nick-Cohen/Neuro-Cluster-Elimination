"""Doc-sync tests: enforce that docs/config_reference.md stays in sync with the schema.

Parses the markdown guide's field tables and cross-checks against
nce.config_schema.NESTED_SECTIONS, DEAD_FIELDS, and _LEGACY_FLAT_FIELDS.

Run:
    pytest tests/test_config_docs.py -v
"""
import re
from pathlib import Path

import pytest

from nce.config_schema import DEAD_FIELDS, NESTED_SECTIONS, _LEGACY_FLAT_FIELDS

GUIDE_PATH = Path(__file__).resolve().parent.parent / "docs" / "config_reference.md"


# ── helpers ──────────────────────────────────────────────────────────

def _read_guide():
    """Return the full text of the config reference guide."""
    assert GUIDE_PATH.exists(), f"Config reference guide not found at {GUIDE_PATH}"
    return GUIDE_PATH.read_text()


def _parse_internal_names_from_section_tables(text):
    """Extract internal field names from the 6 main section field tables.

    Each section table has 5 columns:
        | Readable Name | Internal Name | Type | Default | Purpose |

    We match data rows with exactly 5 pipe-delimited columns where
    the first two are back-ticked code spans. This excludes the
    3-column tables used for Dead Fields, Runtime-Injected, and Legacy.
    """
    names = set()
    # Match rows with ≥5 pipe-separated cells whose first two are `code`.
    # A data row: | `readable` | `internal` | `type` | default | purpose |
    for match in re.finditer(
        r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*`[^`]+`",
        text,
        re.MULTILINE,
    ):
        internal = match.group(2)
        names.add(internal)
    return names


def _collect_schema_internal_names():
    """Collect all unique internal (old) field names from NESTED_SECTIONS."""
    names = set()
    for section_fields in NESTED_SECTIONS.values():
        for fdef in section_fields.values():
            names.add(fdef["old_name"])
    return names


def _extract_section(text, heading):
    """Extract the text under a markdown ## heading until the next ## or EOF."""
    pattern = rf"^##\s+{re.escape(heading)}\s*\n(.*?)(?=\n##\s|\Z)"
    m = re.search(pattern, text, re.DOTALL | re.MULTILINE)
    assert m is not None, f"Section '{heading}' not found in guide"
    return m.group(1)


# ── tests ────────────────────────────────────────────────────────────

class TestSchemaDocSync:
    """Every schema field appears in the docs and vice-versa."""

    def test_every_schema_field_documented(self):
        """Every internal field name in NESTED_SECTIONS must appear in the guide's tables."""
        schema_names = _collect_schema_internal_names()
        doc_names = _parse_internal_names_from_section_tables(_read_guide())

        missing = schema_names - doc_names
        assert not missing, (
            f"Schema fields missing from docs/config_reference.md: "
            f"{sorted(missing)}"
        )

    def test_every_documented_field_in_schema(self):
        """Every internal name in the guide's section tables must exist in the schema."""
        schema_names = _collect_schema_internal_names()
        doc_names = _parse_internal_names_from_section_tables(_read_guide())

        orphans = doc_names - schema_names
        assert not orphans, (
            f"Fields documented in guide but not in schema: "
            f"{sorted(orphans)}"
        )


class TestDeadFieldsDocumented:
    """Dead fields must be documented in the Dead Fields section."""

    def test_dead_fields_documented(self):
        text = _read_guide()
        dead_section = _extract_section(text, "Dead Fields")

        missing = []
        for field_name in DEAD_FIELDS:
            if f"`{field_name}`" not in dead_section:
                missing.append(field_name)

        assert not missing, (
            f"Dead fields missing from 'Dead Fields' section: {sorted(missing)}"
        )


class TestLegacyFieldsDocumented:
    """Legacy flat-only fields must be documented in the Legacy Flat-Only Fields section."""

    def test_legacy_fields_documented(self):
        text = _read_guide()
        legacy_section = _extract_section(text, "Legacy Flat-Only Fields")

        missing = []
        for field_name in _LEGACY_FLAT_FIELDS:
            if f"`{field_name}`" not in legacy_section:
                missing.append(field_name)

        assert not missing, (
            f"Legacy fields missing from 'Legacy Flat-Only Fields' section: "
            f"{sorted(missing)}"
        )
