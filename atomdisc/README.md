# AtomDisc Python Package

The `atomdisc` package contains the reusable modules that power AtomDisc’s four-stage workflow. Each subpackage focuses on a specific layer in the stack—from configuration and models to datasets, tokenization, and evaluation.

## Package Structure

- `config/`
  - Stage-specific configuration dataclasses and defaults (e.g. `stage1.py` provides `Stage1Config` for GNN + VQ training).
- `core/`
  - Foundational utilities and shared logic that do not fit into higher-level modules (placeholder folder for future extensions).
- `data/`
  - Documentation on external datasets and checkpoints. See `data/readme.md` for download links and usage notes.
- `datasets/`
  - Task-specific PyTorch dataset wrappers and collators, used by downstream SFT scripts:
    - `forward_dataset.py`: forward reaction prediction.
    - `retrosynthesis_dataset.py`: retrosynthesis tasks.
    - `reagent_dataset.py`: reagent recommendation.
- `evaluation/`
  - Evaluation helpers and scripts; `sft_evaluator_smiles.py` computes metrics for SMILES-generation tasks.
- `metrics/`
  - Metric implementations independent of evaluation loops. `compute_metrics_smiles.py` contains exact-match, BLEU, edit distance, and fingerprint similarity utilities.
- `models/`
  - Model definitions including the GNN encoder (`gnn.py`) and integrated vector-quantization layers under `vector_quantize_pytorch/`.
- `tokenization/`
  - Molecule tokenization logic:
    - `mol_tokenizer.py`: core SMILES→atom-token conversion pipeline.
    - `mol_tokenizer_property.py`: variant tuned for property-prediction prompts.
- `utils/`
  - General-purpose helper functions shared across scripts, e.g. `gnn_vq_utils.py` for loading checkpoints, seeding, logging, and molecule preprocessing.

## Usage

Import modules directly from the package once AtomDisc is installed (e.g., `pip install -e .`). Example:

```python
from atomdisc.models.gnn import GNN
from atomdisc.utils.gnn_vq_utils import load_gnn_vq_models

# Load pre-trained GNN + VQ
device = "cuda"
gnn, vq = load_gnn_vq_models("path/to/stage1_checkpoint.pth", device)
```

Refer to the repository-level README for the end-to-end training pipeline and to `atomdisc/data/readme.md` for dataset sourcing.
