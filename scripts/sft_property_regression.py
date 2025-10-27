# stage4_sft_lora_prop.py
"""
Stage-4 SFT for molecular property-prediction (HIV / …)
-------------------------------------------------------
• 训练集：先取原始数据 → 10 % 划作验证集(不做任何均衡)  
            → 剩余样本随机采样最多 1 万条 → 对该 1 万条做 minority-oversampling  
• 评估指标：Acc / P / R / F1，且打印验证集中第一条样本 Prompt / Gen / Ref  
• 其余保持与上一版一致
"""
from __future__ import annotations
import os
import json
import math
import random
import argparse
import functools
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

from atomdisc.utils.gnn_vq_utils import set_seed, get_logger, load_gnn_vq_models
from atomdisc.tokenization.mol_tokenizer import convert_text_smiles_to_mol_tokens

# ═══════════════════════════════════════════════
# Dataset & collate
# ═══════════════════════════════════════════════
class MolVQPropertyDataset(Dataset):
    def __init__(self, recs: List[Dict[str, Any]], tok, gnn, vq, dev,
                 max_prompt_len=1024, max_resp_len=16):
        self.recs = recs
        self.tok  = tok
        self.gnn, self.vq, self.dev = gnn, vq, dev
        self.max_prompt_len, self.max_resp_len = max_prompt_len, max_resp_len
        self.gnn.eval(); self.vq.eval()
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
            self.tok.pad_token_id = self.tok.eos_token_id

    @torch.no_grad()
    def _smi2tok(self, smi: str) -> str:
        return convert_text_smiles_to_mol_tokens(f"<smiles>{smi}</smiles>",
                                                 self.gnn, self.vq, self.dev) or ""

    def __len__(self): return len(self.recs)

    def __getitem__(self, idx: int):
        r = self.recs[idx]
        smi, label, instr = r["input"], str(r["output"]).strip(), r["instruction"]
        prompt = (
            f"### Instruction:\n{instr}\n\n"
            f"### Input:\nMolecule (SMILES):\n{smi}\n"
            f"Molecule (Structure):\n{self._smi2tok(smi)}\n\n"
            f"### Response:"
        )
        pt = self.tok(prompt, truncation=True, max_length=self.max_prompt_len, add_special_tokens=False)
        rt = self.tok(label , truncation=True, max_length=self.max_resp_len , add_special_tokens=False)
        ids  = [self.tok.bos_token_id] + pt["input_ids"] + rt["input_ids"] + [self.tok.eos_token_id]
        lbls = [-100]*(1+len(pt["input_ids"])) + rt["input_ids"] + [self.tok.eos_token_id]
        ids_t  = torch.tensor(ids ,dtype=torch.long)
        lbls_t = torch.tensor(lbls,dtype=torch.long)
        return {"input_ids":ids_t,"attention_mask":torch.ones_like(ids_t),
                "labels":lbls_t,"prompt":prompt,"target_text":label,
                "task": r.get("task") or r.get("metadata", {}).get("task", "unknown")}

def collate_fn(batch, pad_id:int):
    ids  = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
    labs = pad_sequence([b["labels"]    for b in batch], batch_first=True, padding_value=-100)
    msk  = ids.ne(pad_id).long()
    return {"input_ids":ids,"attention_mask":msk,"labels":labs,
            "prompt":[b["prompt"] for b in batch],
            "target_text":[b["target_text"] for b in batch],
            "task": [b["task"] for b in batch]}

def evaluate(epoch, model, tok, loader, dev, logger, args):
    if loader is None:
        return
    model.eval()

    # 三个任务桶
    buckets = {
        "homo": {"pred": [], "ref": []},
        "lumo": {"pred": [], "ref": []},
        "homo-lumo-gap": {"pred": [], "ref": []},
    }

    rows = []
    first_printed = False

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval@{epoch}", unit="batch"):
            prompts = batch["prompt"]
            refs_batch = [float(x) for x in batch["target_text"]]
            tasks_batch = batch["task"]

            # 编码 prompt → 生成
            inp = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_seq_length,
            ).to(dev)

            gen_ids = model.generate(
                **inp,
                max_new_tokens=args.max_response_len,
                num_beams=args.num_beams_eval,
                do_sample=args.do_sample_eval,
            )

            # 解码
            gen_texts = [
                tok.decode(g[len(i):], skip_special_tokens=True).strip()
                for g, i in zip(gen_ids, inp["input_ids"])
            ]

            # 正则提取 float
            pred_batch = []
            for txt in gen_texts:
                m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)
                pred_batch.append(float(m.group()) if m else np.nan)

            # 收集全部样例
            for prm, ref, gen, val, task in zip(prompts, refs_batch, gen_texts, pred_batch, tasks_batch):
                task = task.lower()
                if task in buckets:
                    buckets[task]["pred"].append(val)
                    buckets[task]["ref"].append(ref)

                rows.append({
                    "prompt": prm,
                    "reference": ref,
                    "generated": gen,
                    "parsed_float": val,
                    "task": task,
                })

            # 打印首条样例
            if not first_printed and len(gen_texts):
                logger.info("\n===== Eval sample =====\n"
                            f"Prompt:\n{prompts[0]}\n"
                            f"Reference: {refs_batch[0]}\n"
                            f"Generated: {gen_texts[0]}\n"
                            f"Parsed: {pred_batch[0]}\n"
                            f"Task: {tasks_batch[0]}\n"
                            "=======================\n")
                first_printed = True

    # === 每个任务分别计算 RMSE / MAE ===
    for task, buf in buckets.items():
        preds = np.array(buf["pred"], dtype=np.float64)
        refs = np.array(buf["ref"], dtype=np.float64)
        mask = ~np.isnan(preds)

        if mask.sum() == 0:
            logger.warning(f"[Eval] {task:<14}: No valid predictions!")
            continue

        rmse = np.sqrt(np.mean((preds[mask] - refs[mask]) ** 2))
        mae = np.mean(np.abs(preds[mask] - refs[mask]))

        logger.info(f"[Eval] {task:<14}  RMSE = {rmse:.4f}   MAE = {mae:.4f}   "
                    f"({mask.sum()}/{len(preds)} valid)")

    model.train()

    # === 保存 JSONL 文件 ===
    out_dir = Path(args.output_dir) / "eval_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"epoch_{epoch}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"[Eval] ✅ saved {len(rows)} records to {out_path}")
# ═══════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════
def sft_lora(args):
    set_seed(args.seed)
    dev=torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger=get_logger("SFT_Property", Path(args.output_dir)/"train.log")
    logger.info(f"Device={dev}")

    # --- frozen GNN+VQ ---
    if not args.stage1_gnn_vq_checkpoint_path:
        raise ValueError("--stage1_gnn_vq_checkpoint_path is required")
    gnn,vq=load_gnn_vq_models(args.stage1_gnn_vq_checkpoint_path,dev)

    # --- tokenizer & base ---
    if Path(args.stage2_model_path).is_dir():
        tok=AutoTokenizer.from_pretrained(args.stage2_model_path, use_fast=True)
        base=LlamaForCausalLM.from_pretrained(args.stage2_model_path, torch_dtype=torch.bfloat16).to(dev)
    else:
        tok=AutoTokenizer.from_pretrained(args.base_tokenizer_path, use_fast=True)
        base=LlamaForCausalLM.from_pretrained(args.base_llm_model_path, torch_dtype=torch.bfloat16).to(dev)
        extra=["<mol>","</mol>"]+[f"<atom_{i}>" for i in range(args.vq_codebook_size)]
        tok.add_special_tokens({"additional_special_tokens":extra}); base.resize_token_embeddings(len(tok))
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token; tok.pad_token_id=tok.eos_token_id

    # --- fresh LoRA ---
    lcfg=LoraConfig(r=args.lora_r,lora_alpha=args.lora_alpha,lora_dropout=args.lora_dropout,
                    target_modules=[m.strip() for m in args.lora_target_modules.split(',')],
                    bias="none",task_type="CAUSAL_LM")
    model=get_peft_model(base,lcfg).train()

   # --- load raw data ---
    all_recs = []
    for t in [x.strip() for x in args.tasks.split(",") if x.strip()]:
        fp = Path(args.dataset_root) / f"{t}.json"
        if fp.is_file():
            all_recs += json.loads(fp.read_text())
    random.shuffle(all_recs)

    # === 使用 metadata["split"] 来划分 train / val ===
    train_recs = [r for r in all_recs if (r.get("metadata") or {}).get("split") == "train"]
    val_recs   = [r for r in all_recs if (r.get("metadata") or {}).get("split") == "test"]

    logger.info(f"▶️ Loaded {len(train_recs)} train / {len(val_recs)} val samples from metadata split.")

    # --- dataloaders ---
    train_ds = MolVQPropertyDataset(train_recs, tok, gnn, vq, dev, args.max_prompt_len, args.max_response_len)
    val_ds   = MolVQPropertyDataset(val_recs  , tok, gnn, vq, dev, args.max_prompt_len, args.max_response_len)
    collate=functools.partial(collate_fn,pad_id=tok.pad_token_id)
    train_lo=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,
                        num_workers=args.num_workers,collate_fn=collate,drop_last=True)
    val_lo  =DataLoader(val_ds ,batch_size=args.eval_batch_size,shuffle=False,
                        num_workers=args.num_workers,collate_fn=collate)

    # --- optimiser ---
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=args.learning_rate,weight_decay=args.weight_decay)
    steps_ep=math.ceil(len(train_lo)/args.gradient_accumulation_steps)
    tot_steps=steps_ep*args.num_epochs
    sch=get_linear_schedule_with_warmup(opt,int(args.warmup_ratio*tot_steps),tot_steps)

    # --- training ---
    for ep in range(args.num_epochs):
        model.train(); ep_loss=0; opt.zero_grad()
        pbar=tqdm(train_lo,desc=f"Ep {ep+1}/{args.num_epochs}")
        for i,b in enumerate(pbar):
            ids,msk,lbl=b["input_ids"].to(dev),b["attention_mask"].to(dev),b["labels"].to(dev)
            loss=model(input_ids=ids,attention_mask=msk,labels=lbl).loss
            (loss/args.gradient_accumulation_steps).backward(); ep_loss+=loss.item()
            if (i+1)%args.gradient_accumulation_steps==0:
                torch.nn.utils.clip_grad_norm_(params,args.max_grad_norm)
                opt.step(); sch.step(); opt.zero_grad()
            pbar.set_postfix({"CE":f"{loss.item():.4f}"})
        logger.info(f"Ep {ep+1} mean CE = {ep_loss/len(train_lo):.4f}")

        if (ep+1)%args.eval_every_epochs==0:
            evaluate(ep+1,model,tok,val_lo,dev,logger,args)

# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--dataset_root", default="")
    pa.add_argument("--tasks", default="qm9")
    pa.add_argument("--stage1_gnn_vq_checkpoint_path", default="")
    pa.add_argument("--stage2_model_path", default="")
    pa.add_argument("--base_tokenizer_path", default="")
    pa.add_argument("--base_llm_model_path", default="")
    pa.add_argument("--vq_codebook_size", type=int, default=512)
    pa.add_argument("--output_dir", default="./lumo_dataset_out")

    # training
    pa.add_argument("--num_epochs", type=int, default=10)
    pa.add_argument("--batch_size", type=int, default=4)
    pa.add_argument("--eval_batch_size", type=int, default=4)
    pa.add_argument("--gradient_accumulation_steps", type=int, default=6)
    pa.add_argument("--learning_rate", type=float, default=3e-5)
    pa.add_argument("--weight_decay", type=float, default=1e-2)
    pa.add_argument("--warmup_ratio", type=float, default=0.1)
    pa.add_argument("--max_grad_norm", type=float, default=1.0)
    pa.add_argument("--max_prompt_len", type=int, default=2048)
    pa.add_argument("--max_response_len", type=int, default=16)

    # LoRA
    pa.add_argument("--lora_r", type=int, default=8)
    pa.add_argument("--lora_alpha", type=int, default=32)
    pa.add_argument("--lora_dropout", type=float, default=0.05)
    pa.add_argument("--lora_target_modules", default="q_proj,v_proj,k_proj,o_proj")

    # misc
    pa.add_argument("--device", default="cuda")
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--num_workers", type=int, default=0)
    pa.add_argument("--eval_every_epochs", type=int, default=1)

    # generation (eval)
    pa.add_argument("--num_beams_eval", type=int, default=1)
    pa.add_argument("--do_sample_eval", action="store_true")
    pa.add_argument("--max_seq_length", type=int, default=1024)

    sft_lora(pa.parse_args())