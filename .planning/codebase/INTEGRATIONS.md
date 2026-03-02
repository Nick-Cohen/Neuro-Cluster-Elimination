# External Integrations

**Analysis Date:** 2026-02-21

## APIs & External Services

**Probabilistic Graphical Models:**
- pyGMs - Weighted Mini-Bucket inference framework
  - SDK/Client: `pyGMs` package (local editable install at `/home/cohenn1/SDBE/PyGMs`)
  - Usage: Factor graph representation, elimination ordering, message passing
  - Key imports: `from pyGMs import wmb`, `from pyGMs.graphmodel import eliminationOrder`, `from pyGMs.neuro import *`
  - Files: `nce/inference/graphical_model.py`, `nce/inference/bucket.py`, `nce/utils/pygms_conversion.py`, `nce/utils/pygms_wmb_interface.py`

## Data Storage

**Databases:**
- Not detected - No database integrations found

**File Storage:**
- Local filesystem only
  - UAI file format for graphical model definitions (read via `nce/inference/graphical_model.py._load_from_uai()`)
  - Evidence files read via `pyGMs.filetypes.readEvidence14`
  - Model checkpoints stored to disk (training artifacts)
  - Location: Paths specified in configuration dictionaries or test problem definitions

**Caching:**
- In-memory PyTorch tensors during training
- No external caching service detected

## Authentication & Identity

**Auth Provider:**
- Not applicable - No external authentication required

## Monitoring & Observability

**Error Tracking:**
- Not detected - No error tracking service integrated

**Logs:**
- Approach: Direct printing and tqdm progress bars to console
- Files using logging: `nce/neural_networks/train.py`, `nce/inference/graphical_model.py`
- No structured logging framework (no logging module or third-party logger)

## CI/CD & Deployment

**Hosting:**
- Not detected - No cloud hosting configuration

**CI Pipeline:**
- Not detected - No CI/CD configuration files (.github/workflows, .gitlab-ci.yml, etc.)

## Environment Configuration

**Required env vars:**
- Not detected - System uses Python dictionaries for configuration
- Device selection controlled by `device` parameter in config dicts (hardcoded as 'cuda' or 'cpu')
- No environment variables required for runtime

**Secrets location:**
- Not applicable - No secrets management detected

## Webhooks & Callbacks

**Incoming:**
- None detected

**Outgoing:**
- None detected

## Computational Resources

**GPU/CUDA:**
- PyTorch CUDA 11.7 support
  - Device: 'cuda' for GPU, 'cpu' for CPU fallback
  - Examples: `device = 'cuda'` hardcoded in notebooks
  - Memory management: PyTorch garbage collection, manual .to(device) transfers
  - Files: All neural network and inference modules

**Test Problem Data:**
- External benchmark problems loaded at runtime:
  - Files: UAI format problem definitions
  - Sources: `nce/problems.py` (if present) or hardcoded test problems in notebooks
  - Problems include: grid10x10, grid20x20, pedigree, RBM, and others
  - Loading: `test_problems[key]` dictionary access pattern

---

*Integration audit: 2026-02-21*
