import os
import argparse
import logging
import math
import functools
import json
import random
import re
from typing import List, Dict, Any, Tuple

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, LlamaForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

from atomdisc.utils.gnn_vq_utils import (
    get_logger,
    set_seed,
    load_gnn_vq_models,
)
from atomdisc.tokenization.mol_tokenizer import convert_text_smiles_to_mol_tokens
# --- 数据集和Collator定义 ---

class MultiTaskMoleculeDataset(Dataset):
    """一个可以加载并区分多个SFT任务的数据集。"""
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records
        logging.info(f"Initialized MultiTaskMoleculeDataset with {len(self.records)} records.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]

    @staticmethod
    def load_from_multiple_files(file_path_dict: Dict[str, str]) -> "MultiTaskMoleculeDataset":
        """从一个字典加载多个数据文件，并使用字典的键作为任务标识。"""
        all_records = []
        for task_name, file_path in file_path_dict.items():
            logging.info(f"Loading data for task '{task_name}' from {file_path}...")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    train_records = [rec for rec in data if rec.get('metadata', {}).get('split') == 'train']
                    for record in train_records:
                        record['task_identifier'] = task_name
                        all_records.append(record)
                logging.info(f"  -> Loaded {len(train_records)} train records for task '{task_name}'.")
                print(f"-> Loaded {len(train_records)} train records for task '{task_name}'.")
            except Exception as e:
                logging.error(f"Failed to load or process file {file_path}: {e}", exc_info=True)
        
        random.shuffle(all_records)
        return MultiTaskMoleculeDataset(all_records)

class MultiTaskCollator:
    """一个可以根据任务类型，为不同化学任务构建相应Prompt的Collator。"""
    def __init__(self, tokenizer, gnn, vq, device, max_seq_length=2048, use_structure_token=1):
        self.tokenizer = tokenizer
        self.gnn = gnn
        self.vq = vq
        self.device = device
        self.max_seq_length = max_seq_length
        self.use_structure_token = use_structure_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @torch.no_grad()
    def _smiles_to_mol_tokens(self, smiles: str) -> str:
        if not smiles or not isinstance(smiles, str): return ""
        return convert_text_smiles_to_mol_tokens(f"<smiles>{smiles}</smiles>", self.gnn, self.vq, self.device) or ""

    def _build_prompt(self, record: dict) -> Tuple[str, str] | Tuple[None, None]:
        task_name = record.get('task_identifier', '')
        instruction = record.get('instruction', '')

        if 'forward' in task_name:
            reactants = [smi for smi in record.get('input', '').split('.') if smi]
            smiles_section = "\n".join(reactants)
            if self.use_structure_token:
                mol_tokens_section = "\n".join([self._smiles_to_mol_tokens(s) for s in reactants])
                full_prompt = (f"### Instruction:\n{instruction}\n\n### Input:\nMolecules (SMILES):\n{smiles_section}\n\n"
                            f"Structural representations:\n{mol_tokens_section}\n\n### Response:")
            else:
                full_prompt = (f"### Instruction:\n{instruction}\n\n### Input:\nMolecules (SMILES):\n{smiles_section}\n\n### Response:")
            return full_prompt, record.get('output', '')

        elif 'reagent' in task_name:
            parts = record.get('input', '>>').split('>>')
            reactants_str, product_smi = (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else ("", "")
            reactants = [smi for smi in reactants_str.split('.') if smi]
            smiles_section_r = "\n".join(reactants)
            if self.use_structure_token:
                mol_tokens_section_r = "\n".join([self._smiles_to_mol_tokens(s) for s in reactants])
                mol_tokens_section_p = self._smiles_to_mol_tokens(product_smi)
                full_prompt = (f"### Instruction:\n{instruction}\n\n### Input:\nReactants (SMILES):\n{smiles_section_r}\nReactants (Structure):\n{mol_tokens_section_r}\n\n"
                            f"Product (SMILES):\n{product_smi}\nProduct (Structure):\n{mol_tokens_section_p}\n\n### Response:")
            else:
                full_prompt = (f"### Instruction:\n{instruction}\n\n### Input:\nReactants (SMILES):\n{smiles_section_r}\n\n"
                            f"Product (SMILES):\n{product_smi}\n\n### Response:")
            return full_prompt, record.get('output', '').replace('.', '\n')

        elif 'retrosynthesis' in task_name:
            product_smi = record.get('input', '')
            if self.use_structure_token:
                mol_token_seq = self._smiles_to_mol_tokens(product_smi)
                full_prompt = (f"### Instruction:\n{instruction}\n\n### Input:\nProduct (SMILES):\n{product_smi}\nProduct (Structure):\n{mol_token_seq}\n\n### Response:")
            else:
                full_prompt = (f"### Instruction:\n{instruction}\n\n### Input:\nProduct (SMILES):\n{product_smi}\n\n### Response:")
            return full_prompt, record.get('output', '').replace('.', '\n')

        elif 'description_guided' in task_name:
            description = record.get('input', '')
            full_prompt = (f"### Instruction:\n{instruction}\n\n### Input:\nDescription: {description}\n\n### Response:")
            return full_prompt, record.get('output', '')

        else:
            logging.warning(f"Unknown task identifier '{task_name}' found in record. Skipping.")
            return None, None

    def __call__(self, batch_records: list[dict]):
        batch_items = []
        for record in batch_records:
            prompt, target = self._build_prompt(record)
            if prompt is None or target is None or not target: continue

            prompt_tokenized = self.tokenizer(prompt, add_special_tokens=False)
            response_tokenized = self.tokenizer(" " + target, add_special_tokens=False)
            input_ids = [self.tokenizer.bos_token_id] + prompt_tokenized['input_ids'] + response_tokenized['input_ids'] + [self.tokenizer.eos_token_id]
            if len(input_ids) > self.max_seq_length:
                input_ids = input_ids[:self.max_seq_length]
            labels = list(input_ids)
            prompt_len = len(prompt_tokenized['input_ids']) + 1
            if len(labels) > prompt_len:
                labels[:prompt_len] = [-100] * prompt_len
            
            batch_items.append({'input_ids': torch.tensor(input_ids), 'labels': torch.tensor(labels), 'prompt': prompt, 'target_text': target})

        if not batch_items: return None
        
        final_input_ids = pad_sequence([item['input_ids'] for item in batch_items], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        final_labels = pad_sequence([item['labels'] for item in batch_items], batch_first=True, padding_value=-100)
        final_attention_mask = final_input_ids.ne(self.tokenizer.pad_token_id)

        return {'input_ids': final_input_ids, 'attention_mask': final_attention_mask, 'labels': final_labels,
                'prompt': [item['prompt'] for item in batch_items], 'target_text': [item['target_text'] for item in batch_items]}


def log_sample_generation(model, tokenizer, batch_data, device, logger, args):
    model.eval()
    prompt_text = batch_data['prompt'][0]
    reference_text = batch_data['target_text'][0]
    inputs = tokenizer(prompt_text, return_tensors="pt", max_length=args.max_seq_length, truncation=True).to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=args.max_new_tokens_log,
            num_beams=args.num_beams_log, do_sample=args.do_sample_log, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    prompt_len = inputs['input_ids'].shape[1]
    decoded_prediction = tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)
    logger.info(f"\n--- 💡 Training Sample Log ---\nPrompt (Input):\n{prompt_text}\nReference (Target):\n{reference_text}\nPrediction (Generated):\n{decoded_prediction.strip()}\n---------------------------------\n")
    model.train()


def multitask_pretrain_lora(args):
    # --- Setup ---
    set_seed(args.seed)
    effective_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger("MultiTaskLoRAPre-train", os.path.join(args.output_dir, "multitask_pretrain.log"))
    logger.info(f"Effective device: {effective_device}, Args: {args}")

    # --- Load Models ---
    logger.info("Loading prerequisite models...")
    if not args.stage1_gnn_vq_checkpoint_path:
        raise ValueError("--stage1_gnn_vq_checkpoint_path is required.")
    if not args.stage2_model_path:
        raise ValueError("--stage2_model_path is required.")
    must_have = {
        "--caption_guided_data": args.caption_guided_data,
        "--forward_pred_data": args.forward_pred_data,
        "--reagent_pred_data": args.reagent_pred_data,
        "--retrosynthesis_data": args.retrosynthesis_data,
    }
    missing = [flag for flag, value in must_have.items() if not value]
    if missing:
        raise ValueError(f"Missing required dataset paths: {', '.join(missing)}")

    gnn, vq = load_gnn_vq_models(args.stage1_gnn_vq_checkpoint_path, effective_device)
    logger.info(f"Loading base LLM with aligned embeddings from: {args.stage2_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.stage2_model_path, use_fast=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    llm_model = LlamaForCausalLM.from_pretrained(args.stage2_model_path, torch_dtype=torch.bfloat16).to(effective_device)
    logger.info("All prerequisite models loaded.")
    # 【新功能】打印embedding size
    logger.info(f"Loaded LLM embedding size: {llm_model.get_input_embeddings().weight.shape}")

    # --- LoRA Config ---
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none", target_modules=args.lora_target_modules.split(','))
    peft_model = get_peft_model(llm_model, lora_config)
    resume_training = args.resume_ckpt_path and os.path.isdir(args.resume_ckpt_path)

    if resume_training:
        logger.info(f"Resuming from checkpoint: {args.resume_ckpt_path}")
        #peft_model.load_state_dict(torch.load(os.path.join(args.resume_ckpt_path, "adapter_model.safetensors")), strict=False)
        from peft import PeftModel
        peft_model = PeftModel.from_pretrained(llm_model, args.resume_ckpt_path, is_trainable=True)
        tokenizer = AutoTokenizer.from_pretrained(args.resume_ckpt_path, use_fast=True)
    else:
        logger.info("Training from scratch.")

    peft_model.print_trainable_parameters()
    logger.info(f"tokenizer vocab size: {len(tokenizer)}")
    logger.info(f"model embedding size: {llm_model.get_input_embeddings().weight.shape[0]}")
    assert len(tokenizer) == llm_model.get_input_embeddings().weight.shape[0], \
        f"Tokenizer and model embedding vocab size mismatch: {len(tokenizer)} vs {llm_model.get_input_embeddings().weight.shape[0]}"
    # --- Dataset and DataLoader ---
    logger.info("Loading and combining datasets for multi-task training...")
    task_files = {"description_guided_gen": args.caption_guided_data, "forward_pred": args.forward_pred_data,
                  "reagent_pred": args.reagent_pred_data, "retrosynthesis": args.retrosynthesis_data}
    dataset = MultiTaskMoleculeDataset.load_from_multiple_files(task_files)
    collator = MultiTaskCollator(
        tokenizer=tokenizer, gnn=gnn, vq=vq, device=effective_device, 
        max_seq_length=args.max_seq_length,
        use_structure_token=args.use_structure_token  # 新增
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collator, drop_last=True)
    logger.info(f"Multi-task dataloader created with {len(dataset)} total records.")

    # --- Optimizer, Scheduler, and Resume Logic ---
    optimizer = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = (len(dataloader) // args.gradient_accumulation_steps) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_ratio * total_steps), num_training_steps=total_steps)
    start_epoch, global_step, optimizer_step_count = 0, 0, 0
    # Resume logic placeholder

    if resume_training:
        optimizer_path = os.path.join(args.resume_ckpt_path, "optimizer.pt")
        scheduler_path = os.path.join(args.resume_ckpt_path, "scheduler.pt")
        training_state_path = os.path.join(args.resume_ckpt_path, "training_state.json")
        if os.path.isfile(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=effective_device))
        if os.path.isfile(scheduler_path):
            scheduler.load_state_dict(torch.load(scheduler_path, map_location=effective_device))
        if os.path.isfile(training_state_path):
            with open(training_state_path, "r") as f:
                state = json.load(f)
                start_epoch = state.get("epoch", 0)
                global_step = state.get("global_step", 0)
                optimizer_step_count = state.get("optimizer_step_count", 0)
        logger.info(f"Resume state: epoch={start_epoch}, global_step={global_step}, optimizer_step_count={optimizer_step_count}")

    # --- Training Loop ---
    logger.info(f"🚀 Starting Multi-task LoRA Pre-training for {args.num_epochs} epochs...")
    peft_model.train()
    for epoch in range(start_epoch, args.num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Multi-task Epoch {epoch+1}/{args.num_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None: continue
            outputs = peft_model(**{k: v.to(effective_device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']})
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step_count += 1
                global_step += 1 # global_step 最好在 optimizer step 之后更新
                if global_step > 0 and global_step % args.sample_output_steps == 0 and 'prompt' in batch:
                    log_sample_generation(peft_model, tokenizer, batch, effective_device, logger, args)
            
            # 【新功能】更新进度条显示
            progress_bar.set_postfix({
                "loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "OptSteps": optimizer_step_count
            })
            
        logger.info(f"Epoch {epoch+1} finished.")
        if (epoch + 1) % args.save_every_epochs == 0 or (epoch + 1) == args.num_epochs:
            save_path = os.path.join(args.output_dir, f"multitask_lora_epoch_{epoch+1}")
            peft_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info(f"Saved multi-task LoRA checkpoint to {save_path}")
            torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
            with open(os.path.join(save_path, "training_state.json"), "w") as f:
                json.dump({
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "optimizer_step_count": optimizer_step_count
                }, f)
            logger.info("Saved optimizer, scheduler, and training state.")
            
    logger.info("Multi-task LoRA pre-training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3 (Multi-task): Pre-train a general chemical knowledge LoRA adapter.")
    parser.add_argument("--stage1_gnn_vq_checkpoint_path", type=str, default="")
    parser.add_argument("--stage2_model_path", type=str, default="")
    parser.add_argument("--use_structure_token", type=int, default=0, help="1: use special token; 0: do not use")
    parser.add_argument("--output_dir", type=str, default="./stage3_multitask_lora")
    parser.add_argument("--caption_guided_data", type=str, default="")
    parser.add_argument("--forward_pred_data", type=str, default="")
    parser.add_argument("--reagent_pred_data", type=str, default="")
    parser.add_argument("--retrosynthesis_data", type=str, default="")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj,k_proj,o_proj")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_every_epochs", type=int, default=1)
    parser.add_argument("--sample_output_steps", type=int, default=50)
    parser.add_argument("--max_new_tokens_log", type=int, default=256)
    parser.add_argument("--num_beams_log", type=int, default=1)
    parser.add_argument("--do_sample_log", action="store_true")
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=False)
    parser.add_argument("--resume_ckpt_path", type=str, default="")
    args = parser.parse_args()
    multitask_pretrain_lora(args)
