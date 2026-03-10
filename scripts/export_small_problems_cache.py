"""Generate missing .uai.vo and .uai.evid files for all 24 small_problems benchmark instances.

Reads existing .uai.ord files from .model_cache/{category}/ and generates:
  - .uai.vo files: SDBE format (# header + one variable per line)
  - .uai.evid files: "0" for models without evidence (existing evid files are preserved)

Usage:
    python scripts/export_small_problems_cache.py

This script does NOT import from nce to avoid triggering pyGMs catalog network calls.
"""
import os

# Project root = parent of this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CACHE_DIR = os.path.join(PROJECT_ROOT, '.model_cache')

# 24 models from small_problems._MODELS (hardcoded to avoid nce import)
# Format: (catalogue_key, source_iB)
# catalogue_key = "{category}/{model_name}"
_MODELS = [
    # iB10-only problems
    ('alchemy/smokers_20', 10),
    ('bn/BN_3', 10),
    ('bn/BN_5', 10),
    ('bn/BN_7', 10),
    ('bn/BN_10', 10),
    ('bn/BN_11', 10),
    ('segmentation/10_14_s.binary', 10),
    ('segmentation/10_16_s.binary', 10),
    ('segmentation/11_4_s.binary', 10),
    # Problems in both iB10 and iB15 (listed once, using iB10)
    ('bn/BN_1', 10),
    ('promedas/or_chain_10.fg', 10),
    ('segmentation/11_17_s.binary', 10),
    ('objdetect/deer_rescaled_0034.K15.F1.5.model', 10),
    ('objdetect/deer_rescaled_0294.K10.F1.75.model', 10),
    ('grids/grid10x10.f5.wrap', 10),
    # iB15-only problems
    ('bn/BN_2', 15),
    ('bn/BN_8', 15),
    ('bn/BN_9', 15),
    ('objdetect/deer_rescaled_0034.K10.F2.model', 15),
    ('objdetect/deer_rescaled_0034.K15.F1.75.model', 15),
    ('objdetect/deer_rescaled_0034.K20.F1.25.model', 15),
    ('objdetect/deer_rescaled_0034.K20.F1.5.model', 15),
    ('csp/29.wcsp', 15),
    ('csp/404.wcsp', 15),
]


def parse_ord_file(ord_path):
    """Parse a .uai.ord file and return list of variable indices.

    Format: single line with "count var1 var2 ... varN"
    Returns: list of str variable indices (in elimination order)
    """
    with open(ord_path, 'r') as f:
        content = f.read().strip()
    parts = content.split()
    # First element is the count; rest are variable indices
    count = int(parts[0])
    variables = parts[1:]
    if len(variables) != count:
        raise ValueError(
            f"Expected {count} variables in {ord_path}, got {len(variables)}"
        )
    return variables


def write_vo_file(vo_path, variables):
    """Write a .uai.vo file in SDBE format.

    Format:
        #
        var1
        var2
        ...
        varN
    """
    with open(vo_path, 'w') as f:
        f.write('#\n')
        for var in variables:
            f.write(str(var) + '\n')


def write_evid_file(evid_path):
    """Write a .uai.evid file with no evidence (content: "0")."""
    with open(evid_path, 'w') as f:
        f.write('0\n')


def process_model(catalogue_key):
    """Process one model: generate .vo (always) and .evid (if missing).

    Returns dict with status flags: has_uai, has_ord, has_vo, has_evid,
    vo_created, evid_created, error.
    """
    category, model_name = catalogue_key.split('/', 1)
    model_dir = os.path.join(CACHE_DIR, category)

    uai_path = os.path.join(model_dir, model_name + '.uai')
    ord_path = os.path.join(model_dir, model_name + '.uai.ord')
    vo_path = os.path.join(model_dir, model_name + '.uai.vo')
    evid_path = os.path.join(model_dir, model_name + '.uai.evid')

    status = {
        'has_uai': os.path.exists(uai_path),
        'has_ord': os.path.exists(ord_path),
        'has_vo': os.path.exists(vo_path),
        'has_evid': os.path.exists(evid_path),
        'vo_created': False,
        'evid_created': False,
        'error': None,
    }

    # Generate .vo file if missing
    if not status['has_vo']:
        if not status['has_ord']:
            status['error'] = f'Missing .ord file: {ord_path}'
            return status
        try:
            variables = parse_ord_file(ord_path)
            write_vo_file(vo_path, variables)
            status['has_vo'] = True
            status['vo_created'] = True
        except Exception as e:
            status['error'] = f'Error creating .vo: {e}'
            return status

    # Generate .evid file if missing (write "0" = no evidence)
    if not status['has_evid']:
        try:
            write_evid_file(evid_path)
            status['has_evid'] = True
            status['evid_created'] = True
        except Exception as e:
            status['error'] = f'Error creating .evid: {e}'
            return status

    return status


def main():
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Processing {len(_MODELS)} models...\n")

    results = []
    vo_created_count = 0
    evid_created_count = 0
    error_count = 0

    for catalogue_key, _source_ib in _MODELS:
        status = process_model(catalogue_key)
        results.append((catalogue_key, status))
        if status['vo_created']:
            vo_created_count += 1
        if status['evid_created']:
            evid_created_count += 1
        if status['error']:
            error_count += 1

    # Print summary table
    print(f"{'Model':<55} {'UAI':>4} {'ORD':>4} {'VO':>4} {'EVID':>5}  Notes")
    print('-' * 85)

    for catalogue_key, status in results:
        uai_mark = 'YES' if status['has_uai'] else 'NO'
        ord_mark = 'YES' if status['has_ord'] else 'NO'
        vo_mark = 'YES' if status['has_vo'] else 'NO'
        evid_mark = 'YES' if status['has_evid'] else 'NO'

        notes = []
        if status['vo_created']:
            notes.append('vo created')
        if status['evid_created']:
            notes.append('evid created')
        if status['error']:
            notes.append(f'ERROR: {status["error"]}')

        note_str = ', '.join(notes) if notes else ''
        print(f"{catalogue_key:<55} {uai_mark:>4} {ord_mark:>4} {vo_mark:>4} {evid_mark:>5}  {note_str}")

    print('-' * 85)
    print(f"\nSummary:")
    print(f"  .vo files created:   {vo_created_count}")
    print(f"  .evid files created: {evid_created_count}")
    print(f"  Errors:              {error_count}")

    if error_count > 0:
        print("\nERRORS DETECTED - some files were not created")
        return 1

    all_complete = all(
        s['has_uai'] and s['has_ord'] and s['has_vo'] and s['has_evid']
        for _, s in results
    )
    if all_complete:
        print("\nAll 24 models have complete cache files (.uai, .ord, .vo, .evid)")
    else:
        print("\nWARNING: Some models are missing files")
        for catalogue_key, status in results:
            if not (status['has_uai'] and status['has_ord'] and status['has_vo'] and status['has_evid']):
                print(f"  INCOMPLETE: {catalogue_key}")
        return 1

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
