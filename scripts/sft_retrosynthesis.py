'''stage4_sft_lora_retrosynthesis.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Stage-4: Supervised fine-tuning (SFT) for the **retrosynthesis-prediction** task.

Key differences from reagent-prediction:
----------------------------------------------------------
1. **Dataset / collator** :  MolVQRetrosynthesisDataset, collate_fn_retrosyn
2. **Default data path**  :  retrosynthesis json file
3. **Evaluator**          :  evaluate_retrosynthesis_prediction (character-level metrics + fingerprints)

Everything else (LoRA config, optimisation, logging) is kept identical so
that the SFT variants can coexist in the same codebase.
'''  
from __future__ import annotations
import functools
import os
import json
import math
import random
import logging
import argparse
from typing import Dict, List, Any

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, LlamaForCausalLM, get_linear_schedule_with_warmup
from peft import PeftModel, LoraConfig, get_peft_model

from atomdisc.utils.gnn_vq_utils import set_seed, get_logger, load_gnn_vq_models
from atomdisc.tokenization.mol_tokenizer import convert_text_smiles_to_mol_tokens
from atomdisc.evaluation.sft_evaluator_smiles import (
    evaluate_retrosynthesis_prediction as evaluate_prediction,
    log_sample_generation,
)

from MolVQDataset_retrosynthesis import (
    MolVQRetrosynthesisDataset as MolVQDataset,
    collate_fn_retrosyn as user_collate_fn,
)

# -------------------------------------------------------------------------
# Main SFT routine
# -------------------------------------------------------------------------
def sft_lora(args):
    # ---- seed & device ----
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger("Stage4_SFT_LoRA_Retro", os.path.join(args.output_dir, "stage4_sft_lora_retro.log"))
    logger.info(f"Device: {device}; args: {args}")

    # ---- load frozen GNN+VQ ----
    gnn, vq = load_gnn_vq_models(args.stage1_gnn_vq_checkpoint_path, device)

    # ---- load / init tokenizer & base LLM ----
    if os.path.isdir(args.stage2_model_path):
        tokenizer = AutoTokenizer.from_pretrained(args.stage2_model_path, use_fast=True)
        base_llm = LlamaForCausalLM.from_pretrained(args.stage2_model_path, torch_dtype=torch.bfloat16).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer_path, use_fast=True)
        base_llm = LlamaForCausalLM.from_pretrained(args.base_llm_model_path, torch_dtype=torch.bfloat16).to(device)
        logger.info(f"Loaded base LLM from {args.base_llm_model_path}")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---- load / init LoRA ----n    
    if os.path.isdir(args.stage3_lora_adapter_path):
        model = PeftModel.from_pretrained(base_llm, args.stage3_lora_adapter_path, is_trainable=True)
        logger.info("Loaded Stage-3 LoRA adapter → continue fine-tuning.")
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[m.strip() for m in args.lora_target_modules.split(',')],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_llm, lora_cfg)
        logger.info("Initialised new LoRA adapter → SFT.")
    model.train()

    # ---- prepare dataset ----
    with open(args.sft_data_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)
    random.shuffle(data)
    train_recs = [d for d in data if d.get("metadata", {}).get("split") == "train"] or data
    val_recs   = [d for d in data if d.get("metadata", {}).get("split") == "test"]
    if not val_recs and len(train_recs) > 1:
        val_recs = train_recs[-len(train_recs)//20 or 1:]
        train_recs = train_recs[:-len(val_recs)]
    logger.info(f"Dataset ⇒ train {len(train_recs)} / val {len(val_recs)}")

    train_ds = MolVQDataset(train_recs, tokenizer, gnn, vq, device, args.max_prompt_len, args.max_response_len, args.use_structure_token)
    val_ds   = MolVQDataset(val_recs,   tokenizer, gnn, vq, device, args.max_prompt_len, args.max_response_len, args.use_structure_token) if val_recs else None

    collate = functools.partial(user_collate_fn, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate) if val_ds else None

    # ---- optimiser ----
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(optim_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.num_epochs
    sched = get_linear_schedule_with_warmup(optim, int(args.warmup_ratio*total_steps), total_steps)

    '''
    logger.info(f"🚀 Initial Evaluation")
    eval_metrics = evaluate_prediction(0, model, tokenizer, val_loader, device, logger, args.output_dir, args)
    logger.info(f"SFT Epoch 0 Validation Metrics: {str(eval_metrics)}")
    '''
    # ---- training loop ----
    global_step = opt_step = 0
    for epoch in range(args.num_epochs):
        model.train(); epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        optim.zero_grad()
        for b_idx, batch in enumerate(pbar):
            ids  = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)
            loss = model(input_ids=ids, attention_mask=attn, labels=labs).loss
            (loss/args.gradient_accumulation_steps).backward()
            epoch_loss += loss.item(); global_step += 1

            if (b_idx+1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(optim_params, args.max_grad_norm)
                optim.step(); sched.step(); optim.zero_grad(); opt_step += 1

            if global_step % args.sample_output_steps == 0:
                log_sample_generation(model, tokenizer, batch, device, logger, args)
            pbar.set_postfix({"CE": f"{loss.item():.4f}", "OptSteps": opt_step})

        logger.info(f"Epoch {epoch+1} avg CE={epoch_loss/len(train_loader):.4f}")

        if val_loader and (epoch+1) % args.eval_every_epochs == 0:
            metrics = evaluate_prediction(epoch+1, model, tokenizer, val_loader, device, logger, args.output_dir, args)
            logger.info(f"Validation metrics: {metrics}")

        # ---- save ----
        if (epoch+1) % args.save_every_epochs == 0 or (epoch+1)==args.num_epochs:
            tag = "final_sft_lora_adapters" if (epoch+1)==args.num_epochs else f"epoch_{epoch+1}_sft_lora_adapters"
            ckpt = os.path.join(args.output_dir, tag)
            model.save_pretrained(ckpt); tokenizer.save_pretrained(ckpt)
            logger.info(f"Saved adapter to {ckpt}")

    logger.info("Retrosynthesis-prediction SFT complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Stage-4 retrosynthesis-prediction SFT (LoRA)")
    p.add_argument("--sft_data_path", default="")
    p.add_argument("--use_structure_token", type=int, default=1, help="1: use special token; 0: do not use")
    p.add_argument("--stage1_gnn_vq_checkpoint_path", default="")
    #p.add_argument("--stage2_model_path", type=str, default="", help="Path to the Stage 2 LLM with pre-trained embeddings. Leave empty to initialize from a base model.")
    p.add_argument("--stage2_model_path", type=str, default="", help="Path to the Stage 2 LLM with pre-trained embeddings. Leave empty to initialize from a base model.")
    p.add_argument("--stage3_lora_adapter_path", default="")
    p.add_argument("--base_tokenizer_path", default="")
    p.add_argument("--base_llm_model_path", default="")
    p.add_argument("--vq_codebook_size", type=int, default=512)
    p.add_argument("--output_dir", default="./stage4_sft_retrosynthesis")

    # training hp
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_prompt_len", type=int, default=1024)
    p.add_argument("--max_response_len", type=int, default=512)
    # LoRA hp
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", default="q_proj,v_proj,k_proj,o_proj")
    # misc
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--sample_output_steps", type=int, default=200)
    p.add_argument("--eval_every_epochs", type=int, default=1)
    p.add_argument("--save_every_epochs", type=int, default=5)
    # generation params
    p.add_argument("--max_new_tokens_eval", type=int, default=2048)
    p.add_argument("--num_beams_eval", type=int, default=2)
    p.add_argument("--do_sample_eval", action="store_true")
    p.add_argument("--max_new_tokens_log", type=int, default=512)
    p.add_argument("--num_beams_log", type=int, default=1)
    p.add_argument("--do_sample_log", action="store_true")
    p.add_argument("--max_seq_length", type=int, default=2048)
    args = p.parse_args()

    sft_lora(args)
