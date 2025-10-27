"""MolVQReagentDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dataset + collate_fn for **reagent-prediction** task so that it is
100 % compatible with the prompt format used in `MultiTaskCollator`
when `task_identifier` contains the substring "reagent".

Prompt template reproduced here for clarity:
------------------------------------------------------------
### Instruction:
{instruction}

### Input:
Reactants (SMILES):
{smi_r_1}\n{smi_r_2}\n...
Reactants (Structure):
<mol>…</mol> (one per line, same order)

Product (SMILES):
{product_smi}
Product (Structure):
<mol>…</mol>

### Response:
<reagent1>\n<reagent2>\n...
------------------------------------------------------------

The dataset will:
* take `input` in the form  "<reactants> >> <product>"  (standard USPTO style)
* convert every SMILES to structure tokens using frozen GNN+VQ
* build the prompt exactly as above
* treat `output` field as the reagent list ('.'-separated ⇒ replaced by "\n")
* return `input_ids`, `labels`, plus prompt/response strings (for logging)
"""
from __future__ import annotations

import random
import logging
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from atomdisc.tokenization.mol_tokenizer import convert_text_smiles_to_mol_tokens

logger = logging.getLogger("MolVQReagentDataset")


def _reactants_and_product(input_field: str) -> tuple[List[str], str]:
    """Split the "reactants>>product" string into list(reactants) and product."""
    parts = input_field.split(" >> ") if " >> " in input_field else input_field.split(">>")
    if len(parts) != 2:
        return [], ""
    reactants_str, product_smi = parts[0].strip(), parts[1].strip()
    reactants = [s for s in reactants_str.split('.') if s]
    return reactants, product_smi


class MolVQReagentDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer,
        gnn,
        vq,
        device: torch.device,
        max_length_prompt: int = 1024,
        max_length_response: int = 512,
        use_vq_code: int = 1,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.gnn = gnn.eval()
        self.vq = vq.eval()
        self.device = device
        self.max_len_prompt = max_length_prompt
        self.max_len_resp = max_length_response
        self.use_vq_code = use_vq_code
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info("tokenizer.pad_token_id was None -> set to eos_token_id")

    # --------------------------------------------------
    def __len__(self):
        return len(self.records)

    # --------------------------------------------------
    def __getitem__(self, idx):
        max_retries = len(self.records)
        cur_idx = idx
        for _ in range(max_retries):
            rec = self.records[cur_idx % len(self.records)]
            instruction = rec.get("instruction", "Given the reactants and product, propose possible reagents.")
            reactants, product = _reactants_and_product(rec.get("input", ""))
            reagents_str = rec.get("output", "")
            if not reactants or not product or not reagents_str:
                cur_idx = random.randint(0, len(self.records) - 1)
                continue

            # --- structure tokens
            if self.use_vq_code:
                try:
                    with torch.no_grad():
                        reactant_token_lines = [
                            convert_text_smiles_to_mol_tokens(f"<smiles>{s}</smiles>", self.gnn, self.vq, self.device)
                            for s in reactants
                        ]
                        prod_tokens = convert_text_smiles_to_mol_tokens(f"<smiles>{product}</smiles>", self.gnn, self.vq, self.device)
                except Exception:
                    cur_idx = random.randint(0, len(self.records) - 1)
                    continue

                if not prod_tokens or any(t == "" for t in reactant_token_lines):
                    cur_idx = random.randint(0, len(self.records) - 1)
                    continue
                if self.use_vq_code:
                    # --- prompt assembly
                    smiles_section_r = "\n".join(reactants)
                    mol_section_r = "\n".join(reactant_token_lines)
                    full_prompt = (
                        f"### Instruction:\n{instruction}\n\n"
                        f"### Input:\nReactants (SMILES):\n{smiles_section_r}\nReactants (Structure):\n{mol_section_r}\n\n"
                        f"Product (SMILES):\n{product}\nProduct (Structure):\n{prod_tokens}\n\n"
                        f"### Response:"
                )
            else:
                smiles_section_r = "\n".join(reactants)
                full_prompt = (
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Input:\nReactants (SMILES):\n{smiles_section_r}\n\n"
                    f"Product (SMILES):\n{product}\n\n"
                    f"### Response:"
                )
            response_clean = reagents_str.replace(".", "\n")

            # --- tokenization
            prom_tok = self.tokenizer(full_prompt, truncation=True, max_length=self.max_len_prompt, add_special_tokens=False)
            resp_tok = self.tokenizer(response_clean, truncation=True, max_length=self.max_len_resp, add_special_tokens=False)

            bos, eos = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
            input_ids = [bos] + prom_tok["input_ids"] + resp_tok["input_ids"] + [eos]
            labels = [-100] * (1 + len(prom_tok["input_ids"])) + resp_tok["input_ids"] + [eos]

            ids_t = torch.tensor(input_ids, dtype=torch.long)
            lab_t = torch.tensor(labels, dtype=torch.long)

            return {
                "input_ids": ids_t,
                "attention_mask": torch.ones_like(ids_t),
                "labels": lab_t,
                "prompt": full_prompt,
                "response": reagents_str,
                "target_text": reagents_str,
            }

        # all retries failed
        pad = self.tokenizer.pad_token_id
        return {
            "input_ids": torch.tensor([pad]),
            "attention_mask": torch.tensor([0]),
            "labels": torch.tensor([-100]),
            "prompt": "ERROR_PROMPT",
            "response": "error",
            "target_text": "error",
        }


# -----------------------------------------------------------------------------
# Collate fn
# -----------------------------------------------------------------------------

def collate_fn_reagent(batch: list[dict], pad_token_id: int):
    valid = [b for b in batch if b["prompt"] != "ERROR_PROMPT"]
    if not valid:
        return None
    ids   = pad_sequence([b["input_ids"] for b in valid], batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence([b["labels"] for b in valid], batch_first=True, padding_value=-100)
    attn  = ids.ne(pad_token_id).long()
    return {
        "input_ids": ids,
        "attention_mask": attn,
        "labels": labels,
        "prompt": [b["prompt"] for b in valid],
        "response": [b["response"] for b in valid],
        "target_text": [b["target_text"] for b in valid],
    }