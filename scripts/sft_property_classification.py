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
import sys
import json
import math
import random
import argparse
import functools
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
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

    def _extract_subtask(self, instruction: str, rec: Dict[str, Any]) -> str:
        """从指令或记录中提取子任务标识"""
        # 优先使用记录中的 subtask/task 字段
        if "subtask" in rec:
            return str(rec["subtask"])
        if "task" in rec:
            return str(rec["task"])
        
        # 直接使用完整的instruction作为子任务标识
        # 因为相同任务的instruction相同，不同任务的instruction不同
        return instruction.strip()

    def __getitem__(self, idx: int):
        r = self.recs[idx]
        smi, label, instr = r["input"], str(r["output"]).strip(), r["instruction"]
        subtask = self._extract_subtask(instr, r)
        
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
                "labels":lbls_t,"prompt":prompt,"target_text":label,"subtask":subtask}

def collate_fn(batch, pad_id:int):
    ids  = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
    labs = pad_sequence([b["labels"]    for b in batch], batch_first=True, padding_value=-100)
    msk  = ids.ne(pad_id).long()
    return {"input_ids":ids,"attention_mask":msk,"labels":labs,
            "prompt":[b["prompt"] for b in batch],
            "target_text":[b["target_text"] for b in batch],
            "subtask":[b["subtask"] for b in batch]}

# ═══════════════════════════════════════════════
# Scaffold split utility
# ═══════════════════════════════════════════════
def get_scaffold(smiles: str) -> str:
    """从SMILES获取Murcko scaffold"""
    if not RDKIT_AVAILABLE:
        return smiles  # 降级到SMILES本身
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles  # 解析失败，返回原SMILES
        
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return smiles  # scaffold提取失败
            
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        if not scaffold_smiles:  # 空字符串
            return smiles
            
        return scaffold_smiles
    except Exception as e:
        # 如果出现任何异常，返回原SMILES
        return smiles

def scaffold_split(recs: List[Dict[str, Any]], val_ratio: float = 0.1, test_ratio: float = 0.1,
                  seed: int = 42, balanced: bool = True) -> tuple:
    """
    基于scaffold进行数据分割
    Args:
        recs: 数据记录列表，每个记录包含 'input' (SMILES) 字段
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        balanced: True=随机scaffold split（更平衡）; False=确定性split（可能不平衡）
    Returns:
        (train_recs, val_recs, test_recs)
    """
    total = len(recs)
    if total == 0:
        return [], [], []

    if not RDKIT_AVAILABLE:
        # 降级到随机分割
        random.seed(seed)
        random.shuffle(recs)
        n_val = max(1, int(total * val_ratio))
        n_test = max(1, int(total * test_ratio))
        n_val = min(n_val, total - 2) if total >= 3 else n_val
        n_test = min(n_test, total - n_val - 1) if total - n_val >= 2 else n_test
        val_recs = recs[:n_val]
        test_recs = recs[n_val:n_val + n_test]
        train_recs = recs[n_val + n_test:]
        return train_recs, val_recs, test_recs

    # 按scaffold分组
    scaffold_to_indices = defaultdict(list)
    for i, rec in enumerate(recs):
        smiles = rec["input"]
        scaffold = get_scaffold(smiles)
        scaffold_to_indices[scaffold].append(i)

    scaffold_groups = list(scaffold_to_indices.values())

    if balanced:
        random.seed(seed)
        random.shuffle(scaffold_groups)
    else:
        scaffold_groups = sorted(scaffold_groups, key=len, reverse=True)

    target_val = total * val_ratio
    target_test = total * test_ratio
    val_indices, test_indices, train_indices = [], [], []

    for group in scaffold_groups:
        if len(val_indices) + len(group) <= target_val:
            val_indices.extend(group)
        elif len(test_indices) + len(group) <= target_test:
            test_indices.extend(group)
        else:
            train_indices.extend(group)

    # 若训练集为空或比例不足，补回剩余数据
    remaining = set(range(total)) - set(val_indices) - set(test_indices) - set(train_indices)
    train_indices.extend(list(remaining))

    train_recs = [recs[i] for i in train_indices]
    val_recs = [recs[i] for i in val_indices]
    test_recs = [recs[i] for i in test_indices]

    return train_recs, val_recs, test_recs

# ═══════════════════════════════════════════════
# Metrics & evaluation
# ═══════════════════════════════════════════════
def _prf(preds, refs):
    tp=sum(p==r=="yes" for p,r in zip(preds,refs))
    fp=sum(p=="yes" and r=="no"  for p,r in zip(preds,refs))
    fn=sum(p=="no"  and r=="yes" for p,r in zip(preds,refs))
    prec=tp/(tp+fp+1e-8); rec=tp/(tp+fn+1e-8); f1=2*prec*rec/(prec+rec+1e-8)
    acc=sum(p==r for p,r in zip(preds,refs)) / (len(refs) or 1)
    return acc,prec,rec,f1

def extract_auroc_from_log(log_file):
    """提取所有 Macro-AUC，返回最大值（优先），若无则提取 ROC-AUC（兼容老版本）"""
    macro_aucs = []
    micro_aucs = []
    with open(log_file) as f:
        for line in f:
            # 优先提取 Macro-AUC（新版本）
            m_macro = re.search(r"Macro-AUC=([0-9\.]+)", line)
            if m_macro:
                macro_aucs.append(float(m_macro.group(1)))
            
            # 兼容老版本的 ROC-AUC（即 Micro-AUC）
            m_micro = re.search(r"ROC-AUC=([0-9\.]+)", line)
            if m_micro:
                micro_aucs.append(float(m_micro.group(1)))
    
    # 优先返回 Macro-AUC，若无则返回 Micro-AUC
    if macro_aucs:
        return max(macro_aucs)
    elif micro_aucs:
        return max(micro_aucs)
    else:
        return float("nan")

def evaluate(epoch, model, tok, loader, dev, logger, args):
    if loader is None:
        return
    model.eval()
    preds_all, refs_all, scores_all, subtasks_all = [], [], [], []
    first = False

    # 获取 yes/no 的 token ID，取第一个 token 就行（更稳）
    tok_yes = tok.encode("yes", add_special_tokens=False)
    tok_no  = tok.encode("no",  add_special_tokens=False)
    id_yes = tok_yes[0]
    id_no  = tok_no[0]

    # 可选：打印一下实际用的 token（调试用）
    logger.info(f"[Token] yes → {tok.convert_ids_to_tokens([id_yes])[0]} (id={id_yes})")
    logger.info(f"[Token] no  → {tok.convert_ids_to_tokens([id_no])[0]} (id={id_no})")

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval@{epoch}", unit="batch"):
            for prm, ref, subtask in zip(batch["prompt"], batch["target_text"], batch["subtask"]):
                inp = tok(prm, return_tensors="pt", max_length=args.max_seq_length, truncation=True).to(dev)

                # 获取模型输出的 logits，取最后一个 token 的 logits
                out = model(**inp)
                logits = out.logits[0, -1]  # shape: [vocab_size]

                # 获取 "yes"/"no" 概率
                prob = torch.softmax(logits[[id_yes, id_no]], dim=-1)
                p_yes = prob[0].item()
                scores_all.append(p_yes)

                # 使用0.5为阈值做分类
                pred = "yes" if p_yes >= 0.5 else "no"
                preds_all.append(pred)
                refs_all.append(ref.lower())
                subtasks_all.append(subtask)

                if not first:
                    print("\n=== Eval Sample 0 ===")
                    print("Prompt:\n", prm)
                    print("Reference:", ref)
                    print("P(yes):", p_yes)
                    print("Predicted:", pred)
                    print("Subtask:", subtask)
                    print("=====================\n")
                    first = True

    # 指标计算（整体）
    acc, p, r, f1 = _prf(preds_all, refs_all)

    # micro（pooled）AUC（仅作参考）
    y_all = np.array([1 if r == "yes" else 0 for r in refs_all])
    try:
        micro_auc = roc_auc_score(y_all, np.array(scores_all))
    except ValueError:
        micro_auc = float("nan")

    # macro（按子任务）AUC —— 社区口径
    task2y, task2s = defaultdict(list), defaultdict(list)
    for y, s, t in zip(y_all, scores_all, subtasks_all):
        task2y[t].append(y)
        task2s[t].append(s)

    aucs = []
    skipped = 0
    task_details = []
    
    for t in sorted(task2y.keys()):
        y = np.array(task2y[t])
        s = np.array(task2s[t])
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        
        if len(np.unique(y)) < 2:  # 无正或无负：跳过
            skipped += 1
            task_details.append(f"{t}: SKIPPED (pos={n_pos}, neg={n_neg})")
            continue
        
        try:
            task_auc = roc_auc_score(y, s)
            aucs.append(task_auc)
            task_details.append(f"{t}: AUC={task_auc:.4f} (pos={n_pos}, neg={n_neg})")
        except ValueError:
            skipped += 1
            task_details.append(f"{t}: ERROR (pos={n_pos}, neg={n_neg})")
    
    macro_auc = float(np.mean(aucs)) if aucs else float("nan")

    # 详细日志输出
    logger.info(f"[Eval] Overall: Acc={acc:.4f}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}")
    logger.info(f"[Eval] Macro-AUC={macro_auc:.4f} over {len(aucs)} tasks (skipped={skipped}); Micro-AUC={micro_auc:.4f}")
    
    # 打印每个子任务的详情
    logger.info("[Task Details]")
    for detail in task_details:
        logger.info(f"  {detail}")
    
    model.train()

def evaluate_single_task(epoch, model, tok, loader, dev, logger, args):
    """评估单个任务，返回AUC"""
    if loader is None:
        return float("nan")
    
    model.eval()
    preds_all, refs_all, scores_all = [], [], []
    first = False

    # 获取 yes/no 的 token ID
    tok_yes = tok.encode("yes", add_special_tokens=False)
    tok_no  = tok.encode("no",  add_special_tokens=False)
    id_yes = tok_yes[0]
    id_no  = tok_no[0]

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval@{epoch}", unit="batch"):
            for prm, ref, subtask in zip(batch["prompt"], batch["target_text"], batch["subtask"]):
                inp = tok(prm, return_tensors="pt", max_length=args.max_seq_length, truncation=True).to(dev)

                out = model(**inp)
                logits = out.logits[0, -1]

                prob = torch.softmax(logits[[id_yes, id_no]], dim=-1)
                p_yes = prob[0].item()
                scores_all.append(p_yes)

                pred = "yes" if p_yes >= 0.5 else "no"
                preds_all.append(pred)
                refs_all.append(ref.lower())

                if not first:
                    logger.info(f"\n=== Eval Sample 0 ===")
                    logger.info(f"Prompt:\n{prm}")
                    logger.info(f"Reference: {ref}")
                    logger.info(f"P(yes): {p_yes}")
                    logger.info(f"Predicted: {pred}")
                    logger.info("=====================\n")
                    first = True

    # 计算指标
    acc, p, r, f1 = _prf(preds_all, refs_all)

    # 计算AUC
    y_all = np.array([1 if r == "yes" else 0 for r in refs_all])
    try:
        task_auc = roc_auc_score(y_all, np.array(scores_all))
    except ValueError:
        task_auc = float("nan")

    logger.info(f"[Eval@{epoch}] Acc={acc:.4f}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}  AUC={task_auc:.4f}")
    
    model.train()
    return task_auc

# ═══════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════
def sft_lora_single_task(args, task_name, task_train_recs, task_val_recs, gnn, vq, tok, base, oversample=True, version_suffix=""):
    """训练单个任务的LoRA模型"""
    dev=torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 为当前任务创建专用输出目录
    task_dir_name = f"task_{task_name[:30].replace('?', '').replace(' ', '_')}{version_suffix}"
    task_output_dir = Path(args.output_dir) / task_dir_name
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger=get_logger(f"SFT_{task_name[:20]}{version_suffix}", task_output_dir/"train.log")
    logger.info(f"🎯 开始训练任务: {task_name} (oversample={oversample})")
    logger.info(f"📊 训练样本: {len(task_train_recs)}, 验证样本: {len(task_val_recs)}")
    
    # 统计当前任务的标签分布
    train_yes = sum(1 for r in task_train_recs if str(r["output"]).strip().lower() == "yes")
    train_no = len(task_train_recs) - train_yes
    val_yes = sum(1 for r in task_val_recs if str(r["output"]).strip().lower() == "yes")
    val_no = len(task_val_recs) - val_yes
    
    logger.info(f"📈 训练集分布: {train_yes} yes, {train_no} no ({train_yes/(train_yes+train_no):.1%} positive)")
    logger.info(f"📉 验证集分布: {val_yes} yes, {val_no} no ({val_yes/(val_yes+val_no):.1%} positive)")

    # --- fresh LoRA for this task ---
    lcfg=LoraConfig(r=args.lora_r,lora_alpha=args.lora_alpha,lora_dropout=args.lora_dropout,
                    target_modules=[m.strip() for m in args.lora_target_modules.split(',')],
                    bias="none",task_type="CAUSAL_LM")
    
    # ✅ 为每个任务重新加载base model（避免LoRA冲突）
    logger.info("📋 为当前任务加载独立的base model")
    if Path(args.stage2_model_path).is_dir():
        fresh_base=LlamaForCausalLM.from_pretrained(args.stage2_model_path, torch_dtype=torch.bfloat16).to(dev)
    else:
        fresh_base=LlamaForCausalLM.from_pretrained(args.base_llm_model_path, torch_dtype=torch.bfloat16).to(dev)
        if hasattr(base, 'config') and base.config.vocab_size != len(tok):
            fresh_base.resize_token_embeddings(len(tok))
    
    model=get_peft_model(fresh_base,lcfg).train()

    # 直接使用传入的训练和验证数据
    train_recs = task_train_recs
    val_recs = task_val_recs
    
    # 单任务数据过采样（如果需要）
    if oversample and train_recs:
        yes=[r for r in train_recs if str(r["output"]).strip().lower()=="yes"]
        no =[r for r in train_recs if str(r["output"]).strip().lower()=="no"]
        if yes and no:
            if len(yes)<len(no):
                yes= yes * math.ceil(len(no)/len(yes))
            elif len(no)<len(yes):
                no = no  * math.ceil(len(yes)/len(no))
        train_recs = (yes+no) if yes and no else train_recs
        random.shuffle(train_recs)
        logger.info(f"📊 任务内平衡: {len(train_recs)} 样本 (yes {sum(str(r['output']).lower()=='yes' for r in train_recs)}, "
                    f"no {sum(str(r['output']).lower()=='no' for r in train_recs)})") 
    else:
        logger.info(f"📊 不进行过采样: {len(train_recs)} 样本")

    # --- dataloaders ---
    train_ds=MolVQPropertyDataset(train_recs,tok,gnn,vq,dev,args.max_prompt_len,args.max_response_len)
    val_ds  =MolVQPropertyDataset(val_recs ,tok,gnn,vq,dev,args.max_prompt_len,args.max_response_len)
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
    best_auc = float('-inf')  # 记录最佳AUC
    all_aucs = []  # 记录所有epoch的AUC
    
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
            task_auc = evaluate_single_task(ep+1,model,tok,val_lo,dev,logger,args)
            all_aucs.append(task_auc)
            
            # 更新最佳AUC
            if not math.isnan(task_auc) and task_auc > best_auc:
                best_auc = task_auc
                logger.info(f"🎯 新的最佳AUC: {best_auc:.4f} (Epoch {ep+1})")
            
    # 汇总AUC信息
    valid_aucs = [auc for auc in all_aucs if not math.isnan(auc)]
    if valid_aucs:
        max_auc = max(valid_aucs)
        final_auc = all_aucs[-1] if all_aucs else float("nan")  # 最后一个epoch的AUC
        
        logger.info(f"📊 AUC汇总: 最高={max_auc:.4f}, 最终={final_auc:.4f}, 所有epoch={[f'{v:.4f}' for v in valid_aucs]}")
        result_auc = max_auc
    else:
        logger.warning("⚠️ 所有epoch的AUC都是NaN")
        result_auc = float("nan")
    
    # 🚨 关键：清理显存
    logger.info("🧹 清理任务显存...")
    del model, fresh_base, train_ds, val_ds, train_lo, val_lo, opt, sch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 显示清理后的显存使用情况
        memory_allocated = torch.cuda.memory_allocated(dev) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(dev) / 1024**3   # GB
        logger.info(f"💾 显存清理后: 已分配={memory_allocated:.2f}GB, 已保留={memory_reserved:.2f}GB")
    
    return result_auc, task_output_dir

# ═══════════════════════════════════════════════
# Main functions
# ═══════════════════════════════════════════════
def sft_lora_all_tasks(args):
    """主控函数：独立训练所有子任务"""
    set_seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 创建总输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main_logger = get_logger("SFT_AllTasks", Path(args.output_dir)/"main.log")
    main_logger.info(f"🚀 开始独立训练所有子任务...")
    main_logger.info(f"Device={dev}")

    # --- 加载共享组件 ---
    main_logger.info("📚 加载共享组件...")
    
    # frozen GNN+VQ
    gnn, vq = load_gnn_vq_models(args.stage1_gnn_vq_checkpoint_path, dev)
    
    # tokenizer & base
    if Path(args.stage2_model_path).is_dir():
        tok = AutoTokenizer.from_pretrained(args.stage2_model_path, use_fast=True)
        base = LlamaForCausalLM.from_pretrained(args.stage2_model_path, torch_dtype=torch.bfloat16).to(dev)
    else:
        tok = AutoTokenizer.from_pretrained(args.base_tokenizer_path, use_fast=True)
        base = LlamaForCausalLM.from_pretrained(args.base_llm_model_path, torch_dtype=torch.bfloat16).to(dev)
        extra = ["<mol>","</mol>"] + [f"<atom_{i}>" for i in range(args.vq_codebook_size)]
        tok.add_special_tokens({"additional_special_tokens": extra})
        base.resize_token_embeddings(len(tok))
    
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # --- 加载和分割数据 ---
    main_logger.info("📊 加载原始数据...")
    all_recs = []
    for t in [x.strip() for x in args.tasks.split(",") if x.strip()]:
        fp = Path(args.dataset_root) / f"{t}.json"
        if fp.is_file():
            all_recs += json.loads(fp.read_text())
    random.shuffle(all_recs)
    main_logger.info(f"总数据量: {len(all_recs)}")

    # Scaffold分割
    if RDKIT_AVAILABLE and args.use_scaffold_split == 1:
        main_logger.info("🧬 使用 Scaffold Split 进行数据分割...")
        train_recs, val_recs, test_recs = scaffold_split(all_recs, val_ratio=0.1, test_ratio=0.1, seed=args.seed, balanced=args.balanced_scaffold)
        main_logger.info("✅ Scaffold Split 完成")
    else:
        main_logger.info("📝 使用随机分割...")
        n_val = max(1, int(0.1 * len(all_recs)))
        n_test = max(1, int(0.1 * len(all_recs)))
        n_val = min(n_val, len(all_recs) - 2) if len(all_recs) >= 3 else n_val
        n_test = min(n_test, len(all_recs) - n_val - 1) if len(all_recs) - n_val >= 2 else n_test
        val_recs = all_recs[:n_val]
        test_recs = all_recs[n_val:n_val + n_test]
        train_recs = all_recs[n_val + n_test:]

    # --- 按子任务分组 ---
    main_logger.info("🎯 按子任务分组数据...")
    
    def group_by_subtask(recs):
        task_groups = defaultdict(list)
        for r in recs:
            subtask = r.get("subtask") or r.get("task") or r["instruction"].strip()
            task_groups[subtask].append(r)
        return task_groups
    
    train_task_groups = group_by_subtask(train_recs)
    val_task_groups = group_by_subtask(val_recs)
    test_task_groups = group_by_subtask(test_recs)
    
    # 确保训练集和验证集有相同的任务
    common_tasks = set(train_task_groups.keys()) & set(val_task_groups.keys()) & set(test_task_groups.keys())
    main_logger.info(f"发现 {len(common_tasks)} 个共同子任务")
    
    if len(common_tasks) != len(train_task_groups) or len(common_tasks) != len(val_task_groups) or len(common_tasks) != len(test_task_groups):
        main_logger.warning("⚠️ 训练集、验证集或测试集的任务不完全一致！")
    
    # --- 逐个训练子任务 ---
    task_results = {}
    failed_tasks = []
    
    for i, task_name in enumerate(sorted(common_tasks), 1):
        main_logger.info(f"\n🎯 [{i}/{len(common_tasks)}] 开始训练任务: {task_name[:50]}...")
        
        # 显示当前显存使用情况
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(dev) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(dev) / 1024**3   # GB
            main_logger.info(f"💾 任务开始前显存: 已分配={memory_allocated:.2f}GB, 已保留={memory_reserved:.2f}GB")
        
        task_train_recs = train_task_groups[task_name]
        task_val_recs = val_task_groups[task_name]
        task_test_recs = test_task_groups[task_name]
        
        # 检查数据量
        train_yes = sum(1 for r in task_train_recs if str(r["output"]).strip().lower() == "yes")
        train_no = len(task_train_recs) - train_yes
        val_yes = sum(1 for r in task_val_recs if str(r["output"]).strip().lower() == "yes")
        val_no = len(task_val_recs) - val_yes
        test_yes = sum(1 for r in task_test_recs if str(r["output"]).strip().lower() == "yes")
        test_no = len(task_test_recs) - test_yes
        
        main_logger.info(f"训练: {train_yes}+{train_no}={len(task_train_recs)}, 验证: {val_yes}+{val_no}={len(task_val_recs)}, 测试: {test_yes}+{test_no}={len(task_test_recs)}")
        
        # 跳过数据太少或单一类别的任务
        if len(task_train_recs) < 10 or len(task_val_recs) < 3 or len(task_test_recs) < 3:
            main_logger.warning(f"⚠️ 跳过任务 {task_name[:30]}... (数据太少)")
            failed_tasks.append(task_name)
            continue
            
        if min(train_yes, train_no) == 0 or min(val_yes, val_no) == 0 or min(test_yes, test_no) == 0:
            main_logger.warning(f"⚠️ 跳过任务 {task_name[:30]}... (单一类别)")
            failed_tasks.append(task_name)
            continue
        
        try:
            # 训练当前任务的两个版本：oversample 和 no-oversample
            main_logger.info(f"🎆 开始训练两个版本: {task_name[:30]}...")
            
            # 版本1: 使用oversample
            main_logger.info(f"💹 版本1: 使用oversample")
            task_auc_oversample, task_output_dir_oversample = sft_lora_single_task(
                args, task_name, task_train_recs, task_val_recs, gnn, vq, tok, base, 
                oversample=True, version_suffix="_oversample"
            )
            
            # 版本2: 不使用oversample
            main_logger.info(f"💷 版本2: 不使用oversample")
            task_auc_no_oversample, task_output_dir_no_oversample = sft_lora_single_task(
                args, task_name, task_train_recs, task_val_recs, gnn, vq, tok, base, 
                oversample=False, version_suffix="_no_oversample"
            )
            
            # 选择最好的结果
            if math.isnan(task_auc_oversample) and math.isnan(task_auc_no_oversample):
                best_auc = float("nan")
                best_version = "both_failed"
            elif math.isnan(task_auc_oversample):
                best_auc = task_auc_no_oversample
                best_version = "no_oversample"
            elif math.isnan(task_auc_no_oversample):
                best_auc = task_auc_oversample
                best_version = "oversample"
            else:
                if task_auc_oversample >= task_auc_no_oversample:
                    best_auc = task_auc_oversample
                    best_version = "oversample"
                else:
                    best_auc = task_auc_no_oversample
                    best_version = "no_oversample"
            
            task_results[task_name] = best_auc
            main_logger.info(f"✅ 任务完成: {task_name[:30]}...")
            main_logger.info(f"📊 oversample AUC={task_auc_oversample:.4f}, no_oversample AUC={task_auc_no_oversample:.4f}")
            main_logger.info(f"🏆 最佳版本: {best_version}, 最终AUC={best_auc:.4f}")
            
        except Exception as e:
            main_logger.error(f"❌ 任务失败: {task_name[:30]}... Error: {str(e)}")
            import traceback
            main_logger.error(f"❌ 详细错误信息: {traceback.format_exc()}")
            failed_tasks.append(task_name)
            # 异常后也要清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                main_logger.info("🧹 异常后清理显存cache")
            continue
    
    # --- 汇总结果 ---
    main_logger.info(f"\n🎊 所有任务训练完成！")
    main_logger.info(f"成功: {len(task_results)} 个任务")
    main_logger.info(f"失败: {len(failed_tasks)} 个任务")
    
    if task_results:
        valid_aucs = [auc for auc in task_results.values() if not math.isnan(auc)]
        macro_auc = np.mean(valid_aucs) if valid_aucs else float("nan")
        
        main_logger.info(f"\n📊 结果汇总:")
        main_logger.info(f"Macro-AUC: {macro_auc:.4f} (over {len(valid_aucs)} valid tasks)")
        
        # 详细结果
        main_logger.info("\n📋 各任务详细结果:")
        for task_name, auc in sorted(task_results.items(), key=lambda x: x[1], reverse=True):
            short_name = task_name[:50] + "..." if len(task_name) > 50 else task_name
            main_logger.info(f"  {short_name}: AUC={auc:.4f}")
        
        if failed_tasks:
            main_logger.info("\n❌ 失败任务:")
            for task in failed_tasks:
                short_name = task[:50] + "..." if len(task) > 50 else task
                main_logger.info(f"  {short_name}")
        
        return macro_auc
    else:
        main_logger.error("❌ 没有任务成功完成！")
        return float("nan")

# ═══════════════════════════════════════════════
# CLI   
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--dataset_root", default="")
    pa.add_argument("--tasks", default="hiv")
    pa.add_argument("--stage1_gnn_vq_checkpoint_path", default="")
    pa.add_argument("--stage2_model_path", default="")
    pa.add_argument("--base_tokenizer_path", default="")
    pa.add_argument("--base_llm_model_path", default="")
    pa.add_argument("--vq_codebook_size", type=int, default=512)
    pa.add_argument("--output_dir", default="./prop_out_single_task")
    pa.add_argument("--oversample", type=int, default=1, help="1: 任务内做minority oversampling（推荐单任务）; 0: 不做过采样")
    pa.add_argument("--use_scaffold_split", type=int, default=1, help="1: 使用scaffold split进行数据分割（默认）; 0: 使用随机分割")
    pa.add_argument("--balanced_scaffold", type=int, default=1, help="1: 随机scaffold split（更平衡）; 0: 确定性scaffold split（可能不平衡但可重现）")
    
    # Task-level oversampling
    pa.add_argument("--task_level_oversample", type=int, default=1, help="1: 启用任务级别的不平衡处理（推荐）; 0: 禁用")
    pa.add_argument("--min_samples_per_class", type=int, default=10, help="每个任务中每个类别的最小样本数")
    pa.add_argument("--balance_threshold", type=float, default=0.4, help="类别平衡阈值，正例比例低于此值或高于(1-此值)时进行oversample")

    # training
    pa.add_argument("--num_epochs", type=int, default=10)
    pa.add_argument("--batch_size", type=int, default=3)
    pa.add_argument("--eval_batch_size", type=int, default=4)
    pa.add_argument("--gradient_accumulation_steps", type=int, default=8)
    pa.add_argument("--learning_rate", type=float, default=2e-5)
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

    pa.add_argument("--multi_seed", default="12,0,1,2,42", help="逗号分隔的seed列表，比如 '42,123,234,345,456'。会跑完所有seed并输出 mean±std")

    args = pa.parse_args()

    if args.multi_seed:
        seed_list = [int(x) for x in args.multi_seed.split(",") if x.strip()]
        aurocs = []
        task_str = "_".join([t.strip() for t in args.tasks.split(",")])
        for seed in seed_list:
            print(f"\n=== [Seed {seed}] Training... ===\n")
            # 设置seed和输出目录
            args.seed = seed
            args.output_dir = os.path.join(task_str, f"seed_{seed}")

            # 如果输出目录存在，建议清空
            if os.path.exists(args.output_dir):
                shutil.rmtree(args.output_dir)

            # 跑一次主流程
            macro_auc = sft_lora_all_tasks(args)
            print(f"[Seed {seed}] Final Macro-AUC: {macro_auc:.4f}")
            aurocs.append(macro_auc)

        # 统计结果
        aurocs_arr = np.array([x for x in aurocs if not math.isnan(x)])
        print("\n" + "="*60)
        print("🎯 Multi-Seed Training Summary")
        print("="*60)
        print(f"📊 Seeds: {seed_list}")
        print(f"📊 Valid Results: {len(aurocs_arr)}/{len(seed_list)}")
        print(f"📋 All AUC: {[f'{v:.4f}' for v in aurocs]}")
        
        if len(aurocs_arr) > 0:
            mean_auc = np.mean(aurocs_arr)
            std_auc = np.std(aurocs_arr, ddof=1) if len(aurocs_arr) > 1 else 0.0
            max_auc = np.max(aurocs_arr)
            min_auc = np.min(aurocs_arr)
            
            print(f"\n🏆 Final Result: {mean_auc:.4f} ± {std_auc:.4f}")
            print(f"📈 Range: [{min_auc:.4f}, {max_auc:.4f}]")
            
            # 如果只有一个seed，特别提示
            if len(aurocs_arr) == 1:
                print("⚠️  注意: 只有1个有效结果，无法计算标准差")
        else:
            print("❌ 没有有效的结果！")
        
        print("="*60)
    else:
        # 单个seed运行
        sft_lora_all_tasks(args)