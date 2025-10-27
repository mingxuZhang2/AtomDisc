# 文件名: train_projector_on_sft.py
# 描述: 使用SFT数据格式，专门训练一个Projector。
#       训练结束后，将最终的投影结果“烘焙”进LLM的嵌入层，并保存整个LLM模型。

import os
import argparse
import logging
import math
import functools
import json
import random

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from atomdisc.utils.gnn_vq_utils import (
    get_logger,
    load_gnn_vq_models,
    set_seed,
)
from atomdisc.datasets.forward_dataset import MolVQDataset, collate_fn as user_collate_fn
from transformers import AutoTokenizer, LlamaForCausalLM, get_linear_schedule_with_warmup

# 定义Projector模型
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers import LlamaConfig
from torch import nn
import torch.nn.init as init

class Projector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, llm_config_path: str):
        super().__init__()
        llama_config = LlamaConfig.from_pretrained(llm_config_path)
        init_std = llama_config.initializer_range  # typically 0.02
        init_std = 1
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

        for layer in self.model:
            if isinstance(layer, nn.Linear):
                init.normal_(layer.weight, mean=0.0, std=init_std)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

        self.norm = LlamaRMSNorm(output_dim, eps=llama_config.rms_norm_eps)
        self.output_scale = 0.0168  # 也可用 llama_config.initializer_range
        
    def forward(self, x):
        x = self.model(x)
        x = self.norm(x)
        x = x * self.output_scale
        return x

def train_projector(args):
    # -------------------- Setup --------------------
    set_seed(args.seed)
    effective_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger("TrainProjectorSFT", os.path.join(args.output_dir, "train_projector_on_sft.log"))
    logger.info(f"Effective device: {effective_device}, Args: {args}")

    # -------------------- Load Models --------------------
    logger.info("Loading frozen GNN, VQ, and base LLM models.")
    gnn, vq = load_gnn_vq_models(args.stage1_gnn_vq_checkpoint_path, effective_device)
    gnn.eval()
    vq.eval()
    
    codebook_weight = vq.codebook.clone().to(effective_device)
    code_dim = codebook_weight.shape[1]

    tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer_path, use_fast=True)
    llm_model = LlamaForCausalLM.from_pretrained(args.base_llm_model_path, torch_dtype=torch.bfloat16)
    
    special_tokens_to_add = ['<mol>', '</mol>'] + [f'<atom_{i}>' for i in range(args.vq_codebook_size)]
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add, 'pad_token': '[PAD]'})
    
    llm_model.resize_token_embeddings(len(tokenizer))
    input_embeddings_layer = llm_model.get_input_embeddings()
    input_embeddings_layer.weight.requires_grad = False
    
    llm_model = llm_model.to(effective_device)
    
    atom_start_token_id = tokenizer.convert_tokens_to_ids('<atom_0>')
    
    config_source = args.stage2_model_path or args.base_llm_model_path or args.base_tokenizer_path
    if not config_source:
        raise ValueError("Please provide --stage2_model_path or --base_llm_model_path or --base_tokenizer_path")
    projector = Projector(
        input_dim=code_dim,
        output_dim=llm_model.config.hidden_size,
        hidden_dim=args.projector_hidden_dim,
        llm_config_path=config_source
    ).to(effective_device)
    logger.info(f"Projector initialized: maps from {code_dim} to {llm_model.config.hidden_size}.")
    
    trainable_params = sum(p.numel() for p in projector.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters in the projector: {trainable_params:,}")


    # --- 保存逻辑的核心函数 ---
    @torch.no_grad()
    def refresh_and_save_llm(output_path):
        logger.info(f"Refreshing LLM's embedding table for saving to {output_path}...")
        projector.eval()
        
        final_projected_codebook = projector(codebook_weight.to(dtype=projector.model[0].weight.dtype))
        atom_end_token_id = atom_start_token_id + args.vq_codebook_size
        input_embeddings_layer.weight.data[atom_start_token_id:atom_end_token_id] = final_projected_codebook.to(input_embeddings_layer.weight.dtype)
        
        os.makedirs(output_path, exist_ok=True)
        llm_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        logger.info(f"Saved LLM with baked-in embeddings and tokenizer to {output_path}")

    # -------------------- Dataset and DataLoader --------------------
    with open(args.sft_data_path, 'r') as f:
        sft_records_raw = json.load(f)
    
    train_dataset = MolVQDataset(sft_records_raw, tokenizer, gnn, vq, effective_device, args.max_prompt_len, args.max_response_len)
    
    custom_collate = functools.partial(user_collate_fn, pad_token_id=tokenizer.pad_token_id)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate)
    logger.info(f"DataLoader created. Train batches: {len(train_dataloader)}.")
    
    # -------------------- Optimizer and Scheduler --------------------
    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    num_optimizer_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    total_steps = num_optimizer_steps_per_epoch * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_ratio * total_steps), num_training_steps=total_steps)

    # -------------------- 断点恢复训练逻辑 --------------------
    start_epoch = 0
    global_step = 0
    resume_flag = str(args.resume_from_checkpoint).lower() == "true"
    if resume_flag == True:
        ckpt_path = args.resume_ckpt_path
        if ckpt_path and os.path.exists(ckpt_path):
            logger.info(f"Attempting to resume training from checkpoint: {ckpt_path}")
            
            projector_path = os.path.join(ckpt_path, "projector.pt")
            optimizer_path = os.path.join(ckpt_path, "optimizer.pt")

            if os.path.exists(projector_path):
                projector.load_state_dict(torch.load(projector_path, map_location=effective_device))
                logger.info("Projector state loaded successfully.")
                
                if os.path.exists(optimizer_path):
                    state = torch.load(optimizer_path, map_location=effective_device)
                    optimizer.load_state_dict(state['optimizer_state_dict'])
                    scheduler.load_state_dict(state['scheduler_state_dict'])
                    start_epoch = state.get('epoch', 0) + 1
                    global_step = state.get('global_step', 0)
                    logger.info(f"Optimizer and scheduler states loaded. Resuming from epoch {start_epoch}.")
                else:
                    logger.warning(f"Projector loaded, but optimizer state not found at {optimizer_path}. Resuming with fresh optimizer.")
            else:
                logger.warning(f"Checkpoint directory found, but projector.pt not found inside. Starting training from scratch.")
        else:
            logger.warning(f"Resume flag is set, but checkpoint path '{ckpt_path}' not found. Starting training from scratch.")
    else:
        logger.info("Start with a new optimizer\n")

    # -------------------- Training Loop --------------------
    logger.info(f"🚀 Starting Projector Training for {args.num_epochs} epochs, starting from epoch {start_epoch}...")
    llm_model.eval()

    for epoch in range(start_epoch, args.num_epochs):
        projector.train()
        epoch_loss_sum = 0
        train_iterator = tqdm(train_dataloader, desc=f"Projector Training Epoch {epoch+1}/{args.num_epochs}")
        
        for batch_idx, batch in enumerate(train_iterator):
            if batch is None: continue
            
            input_ids = batch["input_ids"].to(effective_device)
            
            base_embeds = input_embeddings_layer(input_ids)
            current_projected_codebook = projector(codebook_weight.to(dtype=projector.model[0].weight.dtype))
            atom_mask = (input_ids >= atom_start_token_id) & (input_ids < atom_start_token_id + args.vq_codebook_size)
            codebook_indices = input_ids[atom_mask] - atom_start_token_id
            
            if codebook_indices.numel() > 0:
                selected_embeds = current_projected_codebook[codebook_indices]
                base_embeds[atom_mask] = selected_embeds.to(base_embeds.dtype)
            
            outputs = llm_model(
                inputs_embeds=base_embeds,
                attention_mask=batch["attention_mask"].to(effective_device),
                labels=batch["labels"].to(effective_device)
            )
            loss = outputs.loss
            
            # 【修改】梯度累积
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            epoch_loss_sum += loss.item() * args.gradient_accumulation_steps
            
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(projector.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            train_iterator.set_postfix({"loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
        
        # 处理epoch末尾不足一个累积步长的梯度
        if (len(train_dataloader) % args.gradient_accumulation_steps != 0):
             if args.max_grad_norm > 0:
                 torch.nn.utils.clip_grad_norm_(projector.parameters(), args.max_grad_norm)
             optimizer.step()
             optimizer.zero_grad()
             global_step += 1

        avg_loss = epoch_loss_sum / len(train_dataloader)
        logger.info(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

        # --- 保存逻辑 ---
        if (epoch + 1) % args.save_every_epochs == 0 or (epoch + 1) == args.num_epochs:
            if args.save_intermediate_checkpoints and (epoch + 1) != args.num_epochs:
                output_path = os.path.join(args.output_dir, f"epoch_{epoch+1}")
            else:
                output_path = os.path.join(args.output_dir, "final_model_with_embeddings")
            
            os.makedirs(output_path, exist_ok=True)
            
            # 保存训练状态
            torch.save(projector.state_dict(), os.path.join(output_path, "projector.pt"))
            torch.save({
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step,
            }, os.path.join(output_path, "optimizer.pt"))
            logger.info(f"Saved training state (projector, optimizer) to {output_path}")
            
            # 保存最终产物（带烘焙嵌入的LLM）
            refresh_and_save_llm(output_path)

    logger.info("Projector training finished. Final model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 (re-imagined): Train a projector on SFT data and save the LLM with baked-in embeddings.")
    
    # Paths
    parser.add_argument("--sft_data_path", type=str, default="", help="Path to the JSON SFT data file.")
    parser.add_argument("--stage1_gnn_vq_checkpoint_path", type=str, default="", help="Path to the trained GNN and VQ checkpoint.")
    parser.add_argument("--base_tokenizer_path", type=str, default="", help="Path to the base LLM's tokenizer.")
    parser.add_argument("--base_llm_model_path", type=str, default="", help="Path to the base LLM model.")
    parser.add_argument("--output_dir", type=str, default="./stage2_embedding", help="Directory to save the final model and logs.")
    
    # 【新功能】添加断点恢复参数
    parser.add_argument("--resume_from_checkpoint", default="True", help="Set this flag to resume from a checkpoint.")
    parser.add_argument("--resume_ckpt_path", type=str, default="", help="Path to the checkpoint directory to resume from.")

    # Model and Tokenizer
    parser.add_argument("--vq_codebook_size", type=int, default=512, help="Size of the VQ codebook.")
    parser.add_argument("--projector_hidden_dim", type=int, default=2048, help="Hidden dimension for the projector MLP.")
    
    # Training Hyperparameters
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=6, help="Number of steps to accumulate gradients before an optimizer step.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Sequence Length
    parser.add_argument("--max_prompt_len", type=int, default=2048)
    parser.add_argument("--max_response_len", type=int, default=256)

    # System
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_every_epochs", type=int, default=1)
    parser.add_argument("--save_intermediate_checkpoints", action="store_true", help="Save intermediate checkpoints.")

    args = parser.parse_args()
    train_projector(args)
