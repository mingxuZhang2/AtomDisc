# Training & Fine-tuning Scripts

This folder contains the entry points for each AtomDisc training stage (codebook, embedding alignment, multi-task LoRA pre-training, downstream SFT). Every script is parameterized via `argparse` and ships with empty/relative defaults for dataset and checkpoint paths—override them from the command line when you launch an experiment.

## Checkpoints

Pre-trained artifacts for all stages are hosted at the AtomDisc Hugging Face repository. Download the assets that match your workflow and supply their local paths via the corresponding CLI flags.

- Stage 1 GNN + VQ codebook checkpoints (`--stage1_gnn_vq_checkpoint_path`)
- Stage 2 embedding alignment / projector outputs (`--stage2_model_path` or baked projector weights)
- Stage 3 multi-task LoRA adapters (`--stage3_lora_adapter_path`)
- Example tokenizer/LLM weights prepared for AtomDisc experiments

You can download the checkpoints from the repository: https://huggingface.co/anonymous041/atomdisc

## Scripts Overview

| Script | Stage | Purpose |
| --- | --- | --- |
| `train_codebook.py` | Stage 1 | Train the GNN encoder and VQ codebook on large SMILES corpora. Requires Stage 1 data (`--train_smiles_path`). |
| `train_embedding.py` | Stage 2 | Align VQ embeddings with a base LLM projector. Needs Stage 1 checkpoint, base tokenizer/model, and SFT-style alignment data. |
| `pretrain_lora.py` | Stage 3 | Multi-task LoRA pre-training across caption-guided, forward/reagent/retro reaction datasets. Requires Stage 1 & Stage 2 outputs plus task datasets. |
| `sft_property_classification.py` | Stage 4 | Supervised fine-tuning on molecular property classification benchmarks (e.g., MoleculeNet). |
| `sft_property_regression.py` | Stage 4 | Fine-tune for property regression (QM9). |
| `sft_forward_prediction.py` | Stage 4 | Fine-tune for forward reaction prediction. |
| `sft_reagent_prediction.py` | Stage 4 | Fine-tune for reagent recommendation. |
| `sft_retrosynthesis.py` | Stage 4 | Fine-tune for retrosynthesis prediction. |

## Usage Pattern

1. Download required checkpoints and base models from the Hugging Face page above, then place them under a local directory of your choice.
2. Prepare dataset paths (or keep defaults to fill in later).
3. Launch the relevant script, pointing the `--stage*_...` flags to the downloaded checkpoints, e.g.:
   ```bash
   python train_embedding.py \
     --stage1_gnn_vq_checkpoint_path /path/to/stage1/gnn_vq.pth \
     --sft_data_path /path/to/mol_instructions.json \
     --base_tokenizer_path /path/to/base_tokenizer \
     --base_llm_model_path /path/to/base_llm \
     --output_dir ./stage2_model
   ```
4. For downstream SFT scripts, supply Stage 1/2/3 checkpoints as needed:
   ```bash
   python sft_forward_prediction.py \
     --sft_data_path /path/to/forward.json \
     --stage1_gnn_vq_checkpoint_path /path/to/stage1/gnn_vq.pth \
     --stage2_model_path /path/to/stage2_model \
     --stage3_lora_adapter_path /path/to/stage3_lora \
     --output_dir ./stage4_forward
   ```

Refer to the repository-level README for more detailed pipeline descriptions and dataset requirements.
