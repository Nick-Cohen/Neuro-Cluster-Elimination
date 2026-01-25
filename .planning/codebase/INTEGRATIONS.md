# External Integrations

**Analysis Date:** 2026-01-25

## APIs & External Services

**Graphical Models Library:**
- pyGMs - Probabilistic graphical models package
  - SDK/Client: `import pyGMs as gm` (imported in `nce/inference/graphical_model.py`)
  - Modules: `pyGMs.Var`, `pyGMs.wmb`, `pyGMs.neuro`, `pyGMs.graphmodel`, `pyGMs.filetypes`
  - Purpose: Variable definitions, weighted mini-bucket operations, elimination order computation, model file I/O

**No external cloud APIs detected** - This is a self-contained research/ML package with no third-party API integrations (no AWS, Azure, Google Cloud, Stripe, etc.)

## Data Storage

**Databases:**
- Not applicable - No database integrations (no SQL, MongoDB, etc.)

**File Storage:**
- Local filesystem only
  - Model files: UAI format (`*.uai`) loaded via `pyGMs.filetypes.readEvidence14()`
  - Pickle files: Serialized models and benchmarks (`.pkl` files in `nce/problems/`)
  - CSV data: Input data files (e.g., `casino.csv` in root)
  - Checkpoints: Models saved during training (written to root directory)

**Caching:**
- None detected - No Redis, Memcached, or other caching layers

## Authentication & Identity

**Auth Provider:**
- Not applicable - No user authentication system
- No API keys or credentials required
- This is a batch processing/research tool with no authentication layer

## Monitoring & Observability

**Error Tracking:**
- None detected - No Sentry, Datadog, or similar error tracking

**Logs:**
- Console logging only via Python `logging` module (imported in `nce/neural_networks/train.py`)
- No structured logging or log aggregation
- Progress tracking via `tqdm.notebook` for training loops
- Custom statistics collection via `nce/utils/stats.py`

**Metrics/Statistics:**
- In-memory statistics gathering: `get_message_stats()` in `nce/utils/stats.py`
- Performance benchmarks stored as pickle files in `nce/problems/` (e.g., `benchmarks_12_4_2025.pkl`)

## CI/CD & Deployment

**Hosting:**
- Not applicable - Local development/research tool
- No cloud deployment infrastructure

**CI Pipeline:**
- Not detected - No GitHub Actions, GitLab CI, Jenkins, or other CI/CD configured
- Development conducted via Jupyter notebooks in `notebooks/` directories

## Environment Configuration

**Required env vars:**
- Not applicable - No environment variables required for core functionality
- Configuration passed as Python dictionaries at runtime

**Secrets location:**
- Not applicable - No credentials or secrets management

## Webhooks & Callbacks

**Incoming:**
- None - Not a service that receives webhooks

**Outgoing:**
- None - No webhook or callback mechanisms

## Data Exchange Formats

**Input Formats:**
- UAI (Universal AI) graphical model files - Read via `pyGMs.filetypes.readEvidence14()`
- Python pickle files (`.pkl`) - For serialized model checkpoints and benchmarks
- CSV files - For tabular data (e.g., `casino.csv`)

**Output Formats:**
- Pickle files - Model serialization
- PNG images - Visualization plots saved from matplotlib
- Text logs - Console output and experiment results

## Dependencies on External Code

**pyGMs Package:**
- Used throughout inference layer in `nce/inference/graphical_model.py` (1809 lines)
- Provides: Graphical model primitives, variable elimination, factor operations
- Version: Not pinned (no requirements.txt specified)

**PyTorch Ecosystem:**
- `torch` (PyTorch core)
- `torch.nn`, `torch.optim`, `torch.utils.data` - NN and training
- No direct integration with TorchHub or model zoos

**Scientific Python Stack:**
- numpy - Array operations
- scipy - Scientific computing
- scikit-learn - Decision tree algorithms

**No external service dependencies** - The system is completely self-contained and can run offline once models/data are available.

---

*Integration audit: 2026-01-25*
