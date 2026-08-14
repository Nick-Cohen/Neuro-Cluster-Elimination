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
        'exact_computation_limit': _field('ecl', default=0),
        'ecl':                     _field('ecl', default=0),
        'i_bound':                 _field('iB', default=0),
        'iB':                      _field('iB', default=0),
        'ib2':                     _field('ib2', default=None),
        'approximation_method':    _field('approximation_method', default='nn'),
        'dope_factors':            _field('dope_factors', default=False),
        'masked_net':              _field('masked_net', default=False),
        'masked_net_lambda':       _field('masked_net_lambda', default=1.0),
        'device':                  _field('device', default='cuda'),
        'deterministic_guard':     _field('deterministic_guard', default=False),
        'neurobe_mode':            _field('neurobe_mode', default=False),
        'use_float64':             _field('use_float64', default=False),
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
        # Relative improvement the NeuroBE patience rule must see before it counts an
        # epoch as an improvement: val < best * (1 - neurobe_es_min_delta).
        # 0.0 reproduces the historical bare `<` comparison exactly. Any value above
        # ~1e-6 makes the stopping epoch immune to last-bit float noise; see
        # notebooks/_August-2026/claude_experiments/21-determinism.md.
        'neurobe_es_min_delta':          _field('neurobe_es_min_delta', default=0.0),
        'use_amp':                       _field('use_amp', default=True),
        'training_time_limit':           _field('training_time_limit', default=None),
    }),
    ('sampling', {
        'sampling_scheme':      _field('sampling_scheme', default='uniform'),
        'proposal_sampling':    _field('proposal_sampling', default=False),
        'proposal_mix':         _field('proposal_mix', default='full'),
        'proposal_temperature': _field('proposal_temperature', default=1.0),
        'common_random_numbers': _field('common_random_numbers', default=True),
        'num_samples':          _field('num_samples', default=_REQUIRED),
        'set_size':           _field('set_size', default=None),
        'val_set':            _field('val_set', default=True),
        'stratify_samples':   _field('stratify_samples', default=False),
        'lower_dim':          _field('lower_dim', default=False),
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
        'bw_iB':                      _field('bw_iB', default=None),
        'bw_ib2':                     _field('bw_ib2', default=None),
        'forward_diff_barrier':       _field('fdb', default=False),
        'fdb':                        _field('fdb', default=False),
    }),
    ('output', {
        'debug':                _field('debug', default=False),
        'display_intermediate': _field('display_intermediate', default=False),
        'track_errors':         _field('track_errors', default=False),
        'error_tracking':       _field('error_tracking', default=False),
        'compute_local_error':  _field('compute_local_error', default=False),
        'time_sample_gen':      _field('time_sample_gen', default=False),
        # --- gamma v2 instrumentation (see nce/utils/gamma_trace.py) --------
        # OFF unless gamma_trace_path is set to a JSONL destination.  When
        # unset, every hook is a dict .get() returning None and the traced
        # code path is identical to the untraced one.
        'gamma_trace_path':       _field('gamma_trace_path', default=None),
        'gamma_trace_per_factor': _field('gamma_trace_per_factor', default=False),
        'gamma_trace_sync':       _field('gamma_trace_sync', default=True),
        'gamma_trace_run_id':     _field('gamma_trace_run_id', default=None),
        'gamma_trace_strategy':   _field('gamma_trace_strategy', default=None),
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
# Width parameter conflict validation
# ===================================================================

def _validate_width_parameter_conflicts(flat):
    """Check for conflicting width threshold specifications before translation.
    
    The new width parameter system (ib2, bw_ib2) is mutually exclusive with
    the traditional separate specification of iB/ecl or bw_iB/bw_ecl.
    
    Users can either:
    - Use width parameters: ib2 (derives both iB and ecl), bw_ib2 (derives bw_iB and bw_ecl)
    - Use traditional parameters: iB and/or ecl separately, bw_iB and/or bw_ecl separately
    
    But cannot mix width parameters with traditional parameters in the same direction.
    
    Args:
        flat: Flat config dict (aliases already resolved, but not translated).
    
    Raises:
        ValueError: If width parameters are mixed with traditional parameters.
    """
    # Forward direction: ib2 is mutually exclusive with iB or ecl
    has_ib2 = 'ib2' in flat and flat['ib2'] is not None
    has_iB = 'iB' in flat and flat['iB'] not in (None, 0)
    has_ecl = 'ecl' in flat and flat['ecl'] not in (None, 0)
    
    if has_ib2 and (has_iB or has_ecl):
        specified = ['ib2']
        if has_iB:
            specified.append('iB')
        if has_ecl:
            specified.append('ecl')
        raise ValueError(
            f"Config specifies both width parameter (ib2) and traditional "
            f"threshold parameters ({', '.join(specified[1:])}). "
            f"Use either ib2 (for binary domains) OR iB/ecl (for general domains), not both."
        )
    
    # Backward direction: bw_ib2 is mutually exclusive with bw_iB or bw_ecl
    # (bw_ib2='fw' is a reference, not a width param — skip conflict check)
    bw_ib2_raw = flat.get('bw_ib2')
    has_bw_ib2 = bw_ib2_raw is not None and not (isinstance(bw_ib2_raw, str) and 'fw' in bw_ib2_raw)
    has_bw_iB = 'bw_iB' in flat and flat['bw_iB'] is not None
    bw_ecl_raw = flat.get('bw_ecl')
    has_bw_ecl = bw_ecl_raw is not None and not (isinstance(bw_ecl_raw, str) and 'fw' in bw_ecl_raw)
    
    if has_bw_ib2 and (has_bw_iB or has_bw_ecl):
        specified = ['bw_ib2']
        if has_bw_iB:
            specified.append('bw_iB')
        if has_bw_ecl:
            specified.append('bw_ecl')
        raise ValueError(
            f"Config specifies both width parameter (bw_ib2) and traditional "
            f"backward threshold parameters ({', '.join(specified[1:])}). "
            f"Use either bw_ib2 (for binary domains) OR bw_iB/bw_ecl (for general domains), not both."
        )


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

    # Mutual exclusivity validation BEFORE translation
    # Check that user didn't specify conflicting width parameters
    _validate_width_parameter_conflicts(flat)

    # Width parameter translation: convert ib2 → iB + ecl, bw_ib2 → bw_iB + bw_ecl
    # Runs after alias resolution and flattening, after conflict validation.
    if 'ib2' in flat and flat['ib2'] is not None:
        flat['iB'] = flat['ib2']
        flat['ecl'] = (2 ** flat['ib2']) - 1
        del flat['ib2']
    
    if 'bw_ib2' in flat and flat['bw_ib2'] is not None:
        bw_ib2_val = flat['bw_ib2']
        if isinstance(bw_ib2_val, str) and 'fw' in bw_ib2_val:
            # bw_ib2='fw' → copy forward ecl/iB to backward
            flat['bw_iB'] = flat.get('iB', 0)
            flat['bw_ecl'] = flat.get('ecl', 0)
        else:
            flat['bw_iB'] = bw_ib2_val
            flat['bw_ecl'] = (2 ** bw_ib2_val) - 1
        del flat['bw_ib2']

    # bw_ecl='fw' → copy forward ecl to backward ecl
    if 'bw_ecl' in flat and isinstance(flat.get('bw_ecl'), str) and 'fw' in flat['bw_ecl']:
        flat['bw_ecl'] = flat.get('ecl', 0)
        if flat.get('bw_iB') is None:
            flat['bw_iB'] = flat.get('iB', 0)

    # Time-based num_epochs: "20m" → num_epochs=999999999, training_time_limit=1200
    # Supports 's' (seconds), 'm' (minutes), 'h' (hours)
    num_epochs_raw = flat.get('num_epochs')
    if isinstance(num_epochs_raw, str):
        suffix = num_epochs_raw[-1].lower()
        value = float(num_epochs_raw[:-1])
        if suffix == 's':
            flat['training_time_limit'] = value
        elif suffix == 'm':
            flat['training_time_limit'] = value * 60
        elif suffix == 'h':
            flat['training_time_limit'] = value * 3600
        else:
            raise ValueError(
                f"Invalid num_epochs time format: '{num_epochs_raw}'. "
                f"Use a number or a time string like '20m', '1h', '300s'."
            )
        flat['num_epochs'] = 999_999_999  # effectively unlimited

    # neurobe_mode expansion: fill in NEUROBE_DEFAULTS for any key
    # not already set by the user. User overrides win.
    # Runs before validation so expanded defaults satisfy required-field checks.
    if flat.get('neurobe_mode'):
        for key, default_value in NEUROBE_DEFAULTS.items():
            if key not in flat:
                flat[key] = default_value

    flat = _validate_flat_config(flat, strict=strict)

    # Populate defaults for any keys not already present.
    # Build internal_name → default from the schema (skip _REQUIRED and
    # duplicate aliases that map to the same internal name).
    seen_internal = set()
    for section_fields in NESTED_SECTIONS.values():
        for field_def in section_fields.values():
            internal = field_def['old_name']
            default = field_def['default']
            if internal not in seen_internal and default is not _REQUIRED and default is not None:
                seen_internal.add(internal)
                if internal not in flat:
                    # Use a copy for mutable defaults (e.g. lists)
                    flat[internal] = list(default) if isinstance(default, list) else default

    return flat
