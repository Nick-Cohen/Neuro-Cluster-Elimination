"""Config schema, validation, and flat/nested translation for NCE.

Pure Python — no torch or pyGMs imports. Independently importable for
testing and documentation generation.

Public API:
    prepare_config(config_dict, strict=True) -> dict
"""
# Field documentation: docs/config_reference.md

import warnings
from collections import OrderedDict


# ===================================================================
# Schema: NESTED_SECTIONS
# ===================================================================
# OrderedDict mapping section name -> dict of readable_name -> {
#   'old_name': internal key that consumer code reads,
#   'default': default value (sentinel _REQUIRED if required),
# }
# Fields where readable name == internal name still get an entry so
# the schema is a single source of truth for all known fields.

_REQUIRED = object()  # sentinel for required fields (no default)


def _field(old_name, default=_REQUIRED):
    """Shorthand for a field definition."""
    return {'old_name': old_name, 'default': default}


NESTED_SECTIONS = OrderedDict([
    ('inference', {
        'iB2':                     _field('ecl', default=0),
        'exact_computation_limit': _field('ecl', default=0),
        'ecl':                     _field('ecl', default=0),
        'i_bound':                 _field('iB', default=0),
        'iB':                      _field('iB', default=0),
        'approximation_method':    _field('approximation_method', default='nn'),
        'dope_factors':            _field('dope_factors', default=False),
        'device':                  _field('device', default='cuda'),
        'neurobe_mode':            _field('neurobe_mode', default=False),
    }),
    ('nn', {
        'hidden_sizes':              _field('hidden_sizes', default=[]),
        'use_linspace_bias':         _field('use_linspace_bias', default=False),
        'use_memorizer':             _field('use_memorizer', default=False),
        'custom_hidden_sizes':       _field('custom_hidden_sizes', default=None),
        'init_with_linear_optimum':  _field('init_with_linear_optimum', default=False),
        'weight_decay':              _field('weight_decay', default=0.0),
        # Decision tree fields (dt_ prefix, only used when approximation_method='dt')
        'num_leaves':                _field('num_leaves', default=None),
        'num_iterations':            _field('num_iterations', default=None),
        'dt_learning_rate':          _field('dt_lr', default=None),
        'dt_lr':                     _field('dt_lr', default=None),
        'dt_momentum':               _field('dt_momentum', default=None),
        'dt_random_seed':            _field('dt_random_seed', default=None),
        'dt_convergence_threshold':  _field('dt_convergence_threshold', default=None),
        'quantization_states':       _field('quantization_states', default=None),
        'activation':                _field('activation', default='tanh'),
    }),
    ('training', {
        'num_epochs':                    _field('num_epochs', default=_REQUIRED),
        'num_epochs_phase2':             _field('num_epochs2', default=0),
        'num_epochs2':                   _field('num_epochs2', default=0),
        'loss_fn':                       _field('loss_fn', default=_REQUIRED),
        'loss_fn_phase2':                _field('loss_fn2', default=None),
        'loss_fn2':                      _field('loss_fn2', default=None),
        'optimizer':                     _field('optimizer', default='adam'),
        'learning_rate':                 _field('lr', default=0.001),
        'lr':                            _field('lr', default=0.001),
        'learning_rate_decay':           _field('lr_decay', default=1.0),
        'lr_decay':                      _field('lr_decay', default=1.0),
        'momentum':                      _field('momentum', default=0.9),
        'batch_size':                    _field('batch_size', default=256),
        'patience':                      _field('patience', default=20),
        'min_learning_rate':             _field('min_lr', default=1e-8),
        'min_lr':                        _field('min_lr', default=1e-8),
        'seed':                          _field('seed', default=42),
        'skip_early_stopping':           _field('skip_early_stopping', default=False),
        'nbe_early_stopping':            _field('nbe_early_stopping', default=False),
        'nbe_warmup_epochs':             _field('nbe_warmup_epochs', default=0),
        'convex_early_stopping':         _field('convex_early_stopping', default=False),
        'convex_patience':               _field('convex_patience', default=20),
        'convex_min_delta':              _field('convex_min_delta', default=1e-8),
        'use_validation_early_stopping': _field('use_validation_early_stopping', default=False),
        'inverse_time_decay_constant':   _field('inverse_time_decay_constant', default=100),
        'lr_schedule':                   _field('lr_schedule', default='none'),
        'lr_schedule_max_lr':            _field('lr_schedule_max_lr', default=None),
        'lr_schedule_eta_min':           _field('lr_schedule_eta_min', default=1e-6),
        'lr_schedule_pct_start':         _field('lr_schedule_pct_start', default=0.1),
        'gradient_clip_norm':            _field('grad_clip_norm', default=None),
        'grad_clip_norm':                _field('grad_clip_norm', default=None),
        'nbe_plateau_threshold':         _field('nbe_plateau_threshold', default=0.1),
        'nbe_plateau_window':            _field('nbe_plateau_window', default=25),
        'nbe_plateau_min_improvement':   _field('nbe_plateau_min_improvement', default=0.01),
        'scaled_mse':                    _field('scaled_mse', default=None),
        'normalization_mode':            _field('normalization_mode', default='logspace_mean'),
        'neurobe_early_stopping':        _field('neurobe_early_stopping', default=False),
        'neurobe_stop_iter':             _field('neurobe_stop_iter', default=2),
        'use_amp':                       _field('use_amp', default=True),
    }),
    ('sampling', {
        'sampling_scheme':   _field('sampling_scheme', default='uniform'),
        'num_samples':       _field('num_samples', default=_REQUIRED),
        'set_size':          _field('set_size', default=None),
        'val_set':           _field('val_set', default=True),
        'stratify_samples':  _field('stratify_samples', default=False),
        'lower_dim':         _field('lower_dim', default=False),
    }),
    ('backward', {
        'use_backward_approximation': _field('use_bw_approx', default=False),
        'use_bw_approx':              _field('use_bw_approx', default=False),
        'populate_backward_factors':  _field('populate_bw_factors', default=False),
        'populate_bw_factors':        _field('populate_bw_factors', default=False),
        'backward_ecl':               _field('bw_ecl', default=None),
        'bw_ecl':                     _field('bw_ecl', default=None),
        'backward_i_bound':           _field('backward_iB', default=None),
        'backward_iB':                _field('backward_iB', default=None),
        'forward_diff_barrier':       _field('fdb', default=False),
        'fdb':                        _field('fdb', default=False),
    }),
    ('output', {
        'debug':                _field('debug', default=False),
        'display_intermediate': _field('display_intermediate', default=False),
        'track_errors':         _field('track_errors', default=False),
        'error_tracking':       _field('error_tracking', default=False),
        'plot_messages':        _field('plot_messages', default=False),
        'traced_losses':        _field('traced_losses', default=[]),
        'gather_message_stats': _field('gather_message_stats', default=False),
        'complexity_limit':     _field('complexity_limit', default=0),
        'log_file':             _field('log_file', default=None),  # Documented in docs/config_reference.md § Output Section
    }),
])

SECTION_NAMES = set(NESTED_SECTIONS.keys())


# ===================================================================
# NeuroBE mode defaults — applied when neurobe_mode=True
# ===================================================================
# These are the NeuroBE-faithful training defaults. When neurobe_mode
# is True in the config, each key is set ONLY if not already present
# (explicit user overrides win).

NEUROBE_DEFAULTS = {
    'approximation_method': 'nn',
    'normalization_mode': 'minmax_01',
    'loss_fn': 'neurobe_weighted_mse',
    'batch_size': 256,
    'lr': 0.001,
    'num_epochs': 500,
    'neurobe_early_stopping': True,
    'neurobe_stop_iter': 2,
    'use_bw_approx': False,
    'populate_bw_factors': False,
    'activation': 'relu',
    'use_amp': False,
    'hidden_sizes': 'neurobe,3',
    'skip_early_stopping': True,
    'nbe_early_stopping': False,
    'lower_dim': True,
    'sampling_scheme': 'all',
    'iB': 25,
    'debug': False,
    'traced_losses': [],
    'optimizer': 'adam',
}


# ===================================================================
# Dead fields — present in benchmark configs but never read from config
# ===================================================================

DEAD_FIELDS = {
    'backward_ecl': "Dead field 'backward_ecl'. Use 'bw_ecl' instead.",
    'num_batches_per_set': (
        "Dead field 'num_batches_per_set'. "
        "This value is computed internally from set_size // batch_size."
    ),
}


# ===================================================================
# Field aliases: readable name -> internal name
# ===================================================================
# These cover the cases where the readable/new name differs from the
# internal/old name. Both names are accepted; the alias is resolved
# to the internal name in the output.

FIELD_ALIASES = {
    'learning_rate': 'lr',
    'exact_computation_limit': 'ecl',
    'iB2': 'ecl',
    'i_bound': 'iB',
    'forward_diff_barrier': 'fdb',
    'backward_ecl_limit': 'bw_ecl',
    'num_epochs_phase2': 'num_epochs2',
    'loss_fn_phase2': 'loss_fn2',
    'learning_rate_decay': 'lr_decay',
    'min_learning_rate': 'min_lr',
    'dt_learning_rate': 'dt_lr',
    'gradient_clip_norm': 'grad_clip_norm',
    'backward_i_bound': 'backward_iB',
    'use_backward_approximation': 'use_bw_approx',
    'populate_backward_factors': 'populate_bw_factors',
}

# Reverse mapping for lookup (internal -> readable)
ALIAS_REVERSE = {v: k for k, v in FIELD_ALIASES.items()}


# ===================================================================
# Build the flat field registry: all known internal field names
# ===================================================================

def _build_all_internal_names():
    """Collect all valid internal (old) field names from the schema."""
    names = set()
    for section_fields in NESTED_SECTIONS.values():
        for readable_name, fdef in section_fields.items():
            names.add(fdef['old_name'])
            names.add(readable_name)
    return names

# Legacy fields not in any section but used by consumer code
_LEGACY_FLAT_FIELDS = {'exact', 'memorizer'}

ALL_KNOWN_FLAT_FIELDS = _build_all_internal_names() | _LEGACY_FLAT_FIELDS


# ===================================================================
# Detection
# ===================================================================

def _is_nested(config):
    """Return True if config appears to be in nested section format.

    A config is nested if any key matches a section name AND its value
    is a dict. This avoids false positives when a flat config happens
    to have a key with the same name as a section (its value would be
    a scalar, not a dict).
    """
    return any(
        key in config and isinstance(config[key], dict)
        for key in SECTION_NAMES
    )


# ===================================================================
# Alias resolution
# ===================================================================

def _resolve_aliases(d):
    """Resolve alias names to internal names in a flat dict.

    Returns a new dict with aliases replaced by their internal names.
    If both alias and internal name are present, the alias value is
    used (last-write wins, but this is an unusual edge case).
    """
    result = {}
    for key, value in d.items():
        internal = FIELD_ALIASES.get(key, key)
        result[internal] = value
    return result


# ===================================================================
# Validation: nested config
# ===================================================================

def validate_nested_config(config):
    """Validate a nested config dict.

    Checks:
    - All top-level keys are known section names or known flat fields
    - Within each section, all fields are known for that section
    - Dead fields are caught even inside sections

    Raises ValueError with section and field context on failure.
    """
    for key, value in config.items():
        if key in SECTION_NAMES:
            if not isinstance(value, dict):
                raise ValueError(
                    f"Section '{key}' must be a dict, got {type(value).__name__}"
                )
            _validate_section_fields(key, value)
        else:
            # Top-level key in a nested config that's not a section name
            # could be a flat field mixed into a nested config — disallow
            raise ValueError(
                f"Unknown section '{key}' in nested config. "
                f"Valid sections: {', '.join(sorted(SECTION_NAMES))}"
            )


def _validate_section_fields(section_name, fields):
    """Validate fields within a single nested section."""
    known_fields = NESTED_SECTIONS[section_name]
    for field_name in fields:
        # Check for dead fields first (specific error message)
        # Dead field at flat level is 'backward_ecl'. In nested backward
        # section, the readable name 'backward_ecl' maps to internal 'bw_ecl'
        # — that's a live alias, not dead. Dead fields only matter at the
        # flat level. But 'num_batches_per_set' could appear in training.
        if field_name in DEAD_FIELDS and field_name not in known_fields:
            raise ValueError(
                f"[{section_name}] {DEAD_FIELDS[field_name]}"
            )
        if field_name not in known_fields:
            raise ValueError(
                f"Unknown field '{field_name}' in section '{section_name}'. "
                f"Valid fields for '{section_name}': "
                f"{', '.join(sorted(set(f for f in known_fields)))}"
            )


# ===================================================================
# Flattening: nested -> flat
# ===================================================================

def flatten_config(nested):
    """Flatten a nested config dict to a flat dict with internal key names.

    Iterates sections, resolves each field's readable name to its internal
    name, and merges all into a single flat dict. Applies defaults for
    missing optional fields.

    Args:
        nested: Validated nested config dict.

    Returns:
        Flat dict with internal key names.
    """
    flat = {}

    # First pass: collect all explicitly-set fields
    for section_name, section_data in nested.items():
        if section_name not in SECTION_NAMES:
            continue  # validated earlier; skip if somehow present
        schema = NESTED_SECTIONS[section_name]
        for field_name, value in section_data.items():
            fdef = schema[field_name]
            internal_name = fdef['old_name']
            flat[internal_name] = value

    return flat


# ===================================================================
# Validation: flat config
# ===================================================================

def _validate_flat_config(flat, strict=True):
    """Validate a flat config dict.

    Checks for dead fields and required fields when approximation_method='nn'.

    Args:
        flat: Flat config dict (aliases already resolved).
        strict: If True, dead fields raise ValueError. If False, dead
                fields emit a warning and are stripped.

    Returns:
        The flat dict, possibly with dead fields removed.
    """
    # Check dead fields
    dead_found = []
    for field_name in list(flat.keys()):
        if field_name in DEAD_FIELDS:
            dead_found.append(field_name)

    if dead_found:
        if strict:
            # Report the first dead field (could report all, but one is enough
            # to fix — the user will hit the next one on the next run)
            field = dead_found[0]
            raise ValueError(DEAD_FIELDS[field])
        else:
            for field in dead_found:
                warnings.warn(
                    DEAD_FIELDS[field],
                    UserWarning,
                    stacklevel=3,
                )
                del flat[field]

    # Check required fields when approximation_method='nn'
    approx = flat.get('approximation_method', 'nn')
    if approx == 'nn':
        _check_required_nn_fields(flat)

    return flat


def _check_required_nn_fields(flat):
    """Check that required fields are present when using NN approximation."""
    required_when_nn = ['loss_fn', 'num_epochs', 'num_samples']
    for field_name in required_when_nn:
        if field_name not in flat:
            raise ValueError(
                f"Missing required field '{field_name}' when "
                f"approximation_method='nn'. This field must be provided."
            )


# ===================================================================
# Public entry point
# ===================================================================

def prepare_config(config_dict, strict=False):
    """Validate and normalize a config dict to a flat internal format.

    Auto-detects whether the input is a flat config (legacy format) or
    a nested config (new sectioned format). Validates, resolves aliases,
    flattens if needed, and returns a plain mutable dict with internal
    key names.

    Args:
        config_dict: Config dict (flat or nested), or None.
        strict: If True (default), dead fields raise ValueError.
                If False, dead fields emit a warning and are stripped.
                Use strict=False for backward compatibility with in-tree
                benchmark configs during the transition period.

    Returns:
        Plain dict with internal key names. All consumer code can read
        from this dict using the old key names (ecl, iB, lr, etc.).

    Raises:
        ValueError: On dead fields (strict mode), unknown fields in
                   nested sections, unknown sections, or missing
                   required fields.
    """
    if config_dict is None:
        return {}

    if not isinstance(config_dict, dict):
        raise TypeError(
            f"Config must be a dict or None, got {type(config_dict).__name__}"
        )

    if not config_dict:
        return {}

    # Make a shallow copy to avoid mutating the caller's dict
    config = dict(config_dict)

    if _is_nested(config):
        # Nested config path: validate sections, then flatten
        validate_nested_config(config)
        flat = flatten_config(config)
    else:
        # Flat config path: resolve aliases
        flat = _resolve_aliases(config)

    # neurobe_mode expansion: fill in NEUROBE_DEFAULTS for any key
    # not already set by the user. User overrides win.
    # Runs before validation so expanded defaults satisfy required-field checks.
    if flat.get('neurobe_mode'):
        for key, default_value in NEUROBE_DEFAULTS.items():
            if key not in flat:
                flat[key] = default_value

    flat = _validate_flat_config(flat, strict=strict)

    return flat
