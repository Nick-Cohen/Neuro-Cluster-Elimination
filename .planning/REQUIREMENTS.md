# Requirements: Experiment Runner Framework

**Defined:** 2026-02-21
**Core Value:** Running a tweaked experiment should be as simple as editing a config file and executing one command

## v1.0 Requirements (Complete)

### Configuration

- [x] **CFG-01**: User can define experiment via YAML config file
- [x] **CFG-02**: Config supports: problem, epochs, bw_ecl values, loss function, architectures, num_runs, gpus
- [x] **CFG-03**: Config file is saved with experiment results for reproducibility

### Execution

- [x] **EXC-01**: Single command runs entire experiment (`python run_experiment.py config.yaml`)
- [x] **EXC-02**: Experiments distribute across specified GPUs (0,1,2,3)
- [x] **EXC-03**: Multiple runs execute when num_runs > 1

### Output Organization

- [x] **OUT-01**: Each experiment gets timestamped folder
- [x] **OUT-02**: Each run gets separate subfolder with results and plots
- [x] **OUT-03**: Averaged results and plots go in `averaged/` subfolder
- [x] **OUT-04**: Summary JSON captures overall metrics

### Plotting

- [x] **PLT-01**: Plots auto-generate after experiment completes
- [x] **PLT-02**: Per-bucket local error plots with symlog y-axis
- [x] **PLT-03**: Per-bucket UKL loss plots
- [x] **PLT-04**: Summary plots (bar chart, heatmap, combined views)
- [x] **PLT-05**: Mean plots across runs when num_runs > 1

## v1.1 Requirements

Requirements for Config & Visualization milestone. Each maps to roadmap phases.

### Config Cleanup

- [ ] **CFG2-01**: Dead config fields from removed/unused code paths are identified and removed
- [ ] **CFG2-02**: Config restructured from flat dict to nested sections (nn, backward, sampling, training, inference, output)
- [ ] **CFG2-03**: Config field names cleaned up for clarity and consistency
- [ ] **CFG2-04**: Comprehensive config documentation guide written (every field explained with type, default, and purpose)
- [ ] **CFG2-05**: Code comments added at config definition sites enforcing doc-sync
- [ ] **CFG2-06**: Config validation updated to match new nested structure with clear error messages

### Visualization

- [ ] **VIZ-01**: FastGM object is picklable with full training state (per-bucket training logs, NN weights, loss histories)
- [ ] **VIZ-02**: Comprehensive logging system writes all training details to a configurable log file path
- [ ] **VIZ-03**: Standalone plotting functions accept FastGM objects (e.g. `plot_learning_curves(fastgm)`)
- [ ] **VIZ-04**: Per-NN learning curve visualization (loss over epochs for individual bucket NNs)
- [ ] **VIZ-05**: Comparison plotting functions accept multiple FastGM objects/logs and plot side-by-side comparisons

### Verification

- [ ] **VER-01**: Existing experiments produce identical results with restructured config (regression test)

## Future Requirements

### Enhanced Configuration

- **CFG-04**: Config inheritance (base config + overrides)
- **CFG-05**: Named experiment presets

### Monitoring

- **MON-01**: Progress output during long-running experiments
- **MON-02**: Notification when experiment completes

### Analysis

- **ANL-01**: Confidence intervals on mean plots
- **ANL-02**: Statistical significance tests between configurations

## Out of Scope

| Feature | Reason |
|---------|--------|
| Interactive dashboard | Adds complexity, post-hoc analysis is sufficient |
| Cloud/cluster submission | Local multi-GPU is sufficient for current needs |
| Hyperparameter search | Manual control preferred for research |
| Real-time plot updates | Post-hoc plotting is fine |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CFG-01 | Phase 1 (v1.0) | Complete |
| CFG-02 | Phase 1 (v1.0) | Complete |
| CFG-03 | Phase 2 (v1.0) | Complete |
| EXC-01 | Phase 1 (v1.0) | Complete |
| EXC-02 | Phase 2 (v1.0) | Complete |
| EXC-03 | Phase 2 (v1.0) | Complete |
| OUT-01 | Phase 3 (v1.0) | Complete |
| OUT-02 | Phase 3 (v1.0) | Complete |
| OUT-03 | Phase 3 (v1.0) | Complete |
| OUT-04 | Phase 3 (v1.0) | Complete |
| PLT-01 | Phase 4 (v1.0) | Complete |
| PLT-02 | Phase 4 (v1.0) | Complete |
| PLT-03 | Phase 4 (v1.0) | Complete |
| PLT-04 | Phase 4 (v1.0) | Complete |
| PLT-05 | Phase 4 (v1.0) | Complete |
| CFG2-01 | Phase 5 (v1.1) | Pending |
| CFG2-02 | Phase 5 (v1.1) | Pending |
| CFG2-03 | Phase 5 (v1.1) | Pending |
| CFG2-04 | Phase 6 (v1.1) | Pending |
| CFG2-05 | Phase 6 (v1.1) | Pending |
| CFG2-06 | Phase 5 (v1.1) | Pending |
| VIZ-01 | Phase 7 (v1.1) | Pending |
| VIZ-02 | Phase 7 (v1.1) | Pending |
| VIZ-03 | Phase 8 (v1.1) | Pending |
| VIZ-04 | Phase 8 (v1.1) | Pending |
| VIZ-05 | Phase 8 (v1.1) | Pending |
| VER-01 | Phase 9 (v1.1) | Pending |

**Coverage:**
- v1.0 requirements: 15 total (all complete)
- v1.1 requirements: 12 total
- Mapped to phases: 12 (100%)
- Unmapped: 0

---
*Requirements defined: 2026-02-21*
*Last updated: 2026-03-10 after v1.1 roadmap creation (phases 5-9)*
