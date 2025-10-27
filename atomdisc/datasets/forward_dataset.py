# 文件名: MolVQDataset_bonds_both.py
# 描述: 为正向反应预测任务定制的Dataset类 (更新了prompt格式)

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import random
import logging

from atomdisc.tokenization.mol_tokenizer import convert_text_smiles_to_mol_tokens


class MolVQDataset(Dataset):
    """
    为正向反应预测任务定制的Dataset。
    Prompt 现在采用统一的 SFT 格式：

    ### Instruction:
    <instruction>

    ### Input:
    Molecules (SMILES):
    <reactant‑1>\n<reactant‑2>...

    Structural representations:
    <mol>...</mol>  (逐行, 与 SMILES 一一对应)

    ### Response:
    <model 需预测的产物 SMILES>
    """

    def __init__(self, records, tokenizer, gnn, vq, device, max_length_prompt=1024, max_length_response=512, use_structure_token=1):
        self.records = records
        self.tokenizer = tokenizer
        self.gnn = gnn
        self.vq = vq
        self.gnn.eval()
        self.vq.eval()
        self.device = device
        self.logger = logging.getLogger("ReactionPredictionDataset")
        self.max_length_prompt = max_length_prompt
        self.max_length_response = max_length_response
        self.use_structure_token = use_structure_token

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.logger.info(f"Tokenizer pad_token_id was None, set to eos_token_id: {self.tokenizer.eos_token_id}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        max_retries = len(self.records) or 1
        original_idx = idx
        for attempt in range(max_retries):
            current_idx = idx % len(self.records)
            rec = self.records[current_idx]

            instruction = rec.get('instruction', "With the provided reactants and reagents, propose potential products:")
            input_smiles_str = rec.get('input')
            product_smiles_str = rec.get('output')

            if not all([input_smiles_str, product_smiles_str]):
                self.logger.warning(f"Record at index {current_idx} is missing fields. Trying new sample.")
                idx = random.randint(0, len(self.records) - 1)
                if attempt == max_retries - 1:
                    return self._get_error_item("Missing required fields.")
                continue

            # --- 1. 分离所有输入的 SMILES ---
            smi_list = [smi for smi in input_smiles_str.split('.') if smi]
            if not smi_list:
                self.logger.warning(f"Input SMILES string is empty for record {current_idx}. Trying new sample.")
                idx = random.randint(0, len(self.records) - 1)
                continue

            # --- 2. 将 SMILES 与 <mol> 序列分别收集 ---
            smiles_lines = []
            mol_token_lines = []
            conversion_success = True
            for smi in smi_list:
                with torch.no_grad():
                    mol_tokens = convert_text_smiles_to_mol_tokens(
                        f"<smiles>{smi}</smiles>", self.gnn, self.vq, self.device
                    )
                if not mol_tokens:
                    self.logger.warning(f"Failed to convert INPUT SMILES '{smi}' for record {current_idx}.")
                    conversion_success = False
                    break

                smiles_lines.append(smi)
                mol_token_lines.append(mol_tokens)

            if not conversion_success:
                self.logger.warning(
                    f"Abandoning record {current_idx} due to conversion failure. Input: '{input_smiles_str}'. Trying new sample."
                )
                idx = random.randint(0, len(self.records) - 1)
                if attempt == max_retries - 1:
                    return self._get_error_item("SMILES conversion failed.")
                continue

            # --- 3. 构建符合统一范式的 Prompt ---
            smiles_section = "\n".join(smiles_lines)
            mol_tokens_section = "\n".join(mol_token_lines)
            if self.use_structure_token:
                full_prompt = (
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Input:\nMolecules (SMILES):\n{smiles_section}\n\n"
                    f"Structural representations:\n{mol_tokens_section}\n\n"
                    f"### Response:"
                )
            else:
                full_prompt = (
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Input:\nMolecules (SMILES):\n{smiles_section}\n\n"
                    f"### Response:"
                )

            # --- 4. 产物 (labels) 仍为原始 SMILES 字符串 ---
            target_text = product_smiles_str

            # --- 5. Tokenize & 创建 Labels ---
            prompt_tokenized = self.tokenizer(
                full_prompt, truncation=True, max_length=self.max_length_prompt, add_special_tokens=False
            )
            prompt_ids_list = prompt_tokenized['input_ids']

            response_tokenized = self.tokenizer(
                target_text, truncation=True, max_length=self.max_length_response, add_special_tokens=False
            )
            response_ids_list = response_tokenized['input_ids']

            bos_token_id, eos_token_id = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
            input_ids_list = [bos_token_id] + prompt_ids_list + response_ids_list + [eos_token_id]
            prompt_len_for_labels = 1 + len(prompt_ids_list)  # +1 是 BOS

            input_ids = torch.tensor(input_ids_list, dtype=torch.long)
            labels = torch.full_like(input_ids, -100)

            start_of_response_for_labels = prompt_len_for_labels
            end_of_response_for_labels = prompt_len_for_labels + len(response_ids_list) + 1  # +1 是 EOS
            labels[start_of_response_for_labels:end_of_response_for_labels] = input_ids[
                start_of_response_for_labels:end_of_response_for_labels
            ].clone()

            return {
                'input_ids': input_ids,
                'attention_mask': torch.ones_like(input_ids, dtype=torch.long),
                'labels': labels,
                'prompt_len': prompt_len_for_labels,
                'prompt': full_prompt,
                'response': product_smiles_str,
                'target_text': target_text
            }

        self.logger.error(f"Failed after max retries for original index {original_idx}.")
        return self._get_error_item("Exceeded max retries in __getitem__.")

    def _get_error_item(self, message="Error"):
        """返回一个表示错误的虚拟样本，以便 collate_fn 可以安全地过滤掉它。"""
        return {
            "prompt": "ERROR_PROMPT",
            "input_ids": torch.tensor([self.tokenizer.pad_token_id]),
            "attention_mask": torch.tensor([0]),
            "labels": torch.tensor([-100]),
            "prompt_len": 0,
            "response": message,
            "target_text": message,
        }


def collate_fn(batch, pad_token_id):
    valid_batch_items = [item for item in batch if item and item.get("prompt") != "ERROR_PROMPT"]

    if not valid_batch_items:
        logging.getLogger("collate_fn_SFT").error("Collate_fn received an empty or all-error batch.")
        return None

    input_ids = [item["input_ids"] for item in valid_batch_items]
    labels = [item["labels"] for item in valid_batch_items]

    prompts = [item["prompt"] for item in valid_batch_items]
    responses = [item["response"] for item in valid_batch_items]
    target_texts = [item["target_text"] for item in valid_batch_items]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = input_ids_padded.ne(pad_token_id).long()

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "labels": labels_padded,
        "prompt": prompts,
        "response": responses,
        "target_text": target_texts
    }
