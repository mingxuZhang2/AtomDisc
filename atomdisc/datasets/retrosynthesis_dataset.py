import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import logging
import random

from atomdisc.tokenization.mol_tokenizer import convert_text_smiles_to_mol_tokens


class MolVQRetrosynthesisDataset(Dataset):
    """
    Dataset class for retrosynthesis prediction.
    - Input: single product SMILES + its structure tokens
    - Output: precursor SMILES (possibly multiple, separated by '.')
    - Prompt format conforms to the SFT instruction style.
    """
    def __init__(self, records, tokenizer, gnn, vq, device, max_length_prompt=1024, max_length_response=512, use_vq_code = 1):
        self.records = records
        self.tokenizer = tokenizer
        self.gnn = gnn
        self.vq = vq
        self.device = device
        self.max_length_prompt = max_length_prompt
        self.max_length_response = max_length_response
        self.use_vq_code = use_vq_code
        self.gnn.eval()
        self.vq.eval()
        self.logger = logging.getLogger("RetrosynthesisDataset")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.logger.info("Set pad_token_id to eos_token_id.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        max_retries = len(self.records)
        for attempt in range(max_retries):
            record = self.records[idx % len(self.records)]

            instruction = record.get("instruction", "Given the product molecule, propose possible precursors.")
            product_smi = record.get("input", "")
            precursors_smi = record.get("output", "")
            
            if self.use_vq_code:
                if not product_smi or not precursors_smi:
                    self.logger.warning(f"Missing input/output in record {idx}. Retrying.")
                    idx = random.randint(0, len(self.records) - 1)
                    continue

                # Structure conversion
                with torch.no_grad():
                    mol_token_seq = convert_text_smiles_to_mol_tokens(f"<smiles>{product_smi}</smiles>", self.gnn, self.vq, self.device)
                if not mol_token_seq:
                    self.logger.warning(f"Mol token conversion failed for product: {product_smi}")
                    idx = random.randint(0, len(self.records) - 1)
                    continue

                # Build prompt
                full_prompt = (
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Input:\nProduct (SMILES):\n{product_smi}\n"
                    f"Product (Structure):\n{mol_token_seq}\n\n"
                    f"### Response:"
                )
            else:
                full_prompt = (
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Input:\nProduct (SMILES):\n{product_smi}\n"
                    f"### Response:"
                )
            # Tokenization
            prompt_tokenized = self.tokenizer(full_prompt, truncation=True, max_length=self.max_length_prompt, add_special_tokens=False)
            response_tokenized = self.tokenizer(precursors_smi.replace(".", "\n"), truncation=True, max_length=self.max_length_response, add_special_tokens=False)

            bos, eos = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
            input_ids = [bos] + prompt_tokenized["input_ids"] + response_tokenized["input_ids"] + [eos]
            labels = [-100] * (1 + len(prompt_tokenized["input_ids"])) + response_tokenized["input_ids"] + [eos]

            input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
            labels_tensor = torch.tensor(labels, dtype=torch.long)

            return {
                "input_ids": input_ids_tensor,
                "attention_mask": torch.ones_like(input_ids_tensor),
                "labels": labels_tensor,
                "prompt": full_prompt,
                "response": precursors_smi,
                "target_text": precursors_smi
            }

        return self._get_error_item("Max retries exceeded.")

    def _get_error_item(self, message="Error"):
        return {
            "input_ids": torch.tensor([self.tokenizer.pad_token_id]),
            "attention_mask": torch.tensor([0]),
            "labels": torch.tensor([-100]),
            "prompt": "ERROR_PROMPT",
            "response": message,
            "target_text": message
        }


def collate_fn_retrosyn(batch, pad_token_id):
    valid = [b for b in batch if b["prompt"] != "ERROR_PROMPT"]
    if not valid:
        return None

    input_ids = pad_sequence([item["input_ids"] for item in valid], batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence([item["labels"] for item in valid], batch_first=True, padding_value=-100)
    attention_mask = input_ids.ne(pad_token_id).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt": [item["prompt"] for item in valid],
        "response": [item["response"] for item in valid],
        "target_text": [item["target_text"] for item in valid]
    }
