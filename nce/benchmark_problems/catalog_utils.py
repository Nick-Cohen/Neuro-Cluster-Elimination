"""Utilities for initializing and accessing the pyGMs UAI model catalogue."""
import os
import json
from pyGMs.data.catalog import Catalog

# Default cache location (project-level, gitignored)
_DEFAULT_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    '.model_cache'
)
_SOURCE_URL = 'https://ics.uci.edu/~ihler/uai-data/models/index.json'

# Additional model sets not in the default remote index
_EXTRA_SETS = {
    "pedigree": {
        "name": "pedigree",
        "description": "Pedigree genetic network models from UAI competitions",
        "modelset": "https://ics.uci.edu/~ihler/uai-data/models/pedigree/statistics.csv",
        "types": ["uai"]
    }
}


def _ensure_extra_sets(cache_dir):
    """Ensure extra model sets (e.g. pedigree) are in the cache index."""
    index_path = os.path.join(cache_dir, 'index.json')
    if not os.path.exists(index_path):
        return  # Will be created by Catalog.set_cache
    with open(index_path) as f:
        idx = json.load(f)
    updated = False
    for key, entry in _EXTRA_SETS.items():
        if key not in idx:
            idx[key] = entry
            updated = True
    if updated:
        with open(index_path, 'w') as f:
            json.dump(idx, f, indent=4)


def get_catalog(cache_dir=None, refresh=False):
    """Get a configured Catalog instance with all model sets available.

    Args:
        cache_dir: Path to cache directory. Defaults to .model_cache in project root.
        refresh: If True, force re-download of the catalogue index.

    Returns:
        Catalog instance ready to access models.
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE
    os.makedirs(cache_dir, exist_ok=True)
    c = Catalog(cache=cache_dir, source=_SOURCE_URL)
    if refresh:
        c.update_cache()
    _ensure_extra_sets(cache_dir)
    return c
