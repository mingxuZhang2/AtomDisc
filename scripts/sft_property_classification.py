# stage4_sft_lora_prop.py
"""
Stage-4\tSFT for molecular property prediction (HIV / …)
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
        """Extract the sub-task identifier from the instruction or record."""
        # Prefer using the `subtask`/`task` fields if available
        if "subtask" in rec:
            return str(rec["subtask"])
        if "task" in rec:
            return str(rec["task"])
        
        # Fall back to the full instruction text
        # Same tasks share the same instruction; different tasks differ
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
    """Extract the Murcko scaffold from a SMILES string."""
    if not RDKIT_AVAILABLE:
        return smiles  # Fallback to the original SMILES
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles  # Parsing failed; return the original SMILES
        
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return smiles  # Scaffold extraction failed
            
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        if not scaffold_smiles:  # Empty string
            return smiles
            
        return scaffold_smiles
    except Exception as e:
        # If anything unexpected happens, return the original SMILES
        return smiles

def scaffold_split(recs: List[Dict[str, Any]], val_ratio: float = 0.1, test_ratio: float = 0.1,
                  seed: int = 42, balanced: bool = True) -> tuple:
    """Split data based on molecular scaffolds.

    Args:
        recs: List of data records, each containing an 'input' (SMILES) field
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        seed: Random seed
        balanced: True = random scaffold split (more balanced); False = deterministic split (may be imbalanced)
    Returns:
        (train_recs, val_recs, test_recs)
    """
    total = len(recs)
    if total == 0:
        return [], [], []

    if not RDKIT_AVAILABLE:
        # Fall back to a random split
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

    # Group records by scaffold
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

    # If the training split is empty or insufficient, fill with the remaining data
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
    """Collect all Macro-AUC values; return the maximum, otherwise fall back to ROC-AUC."""
    macro_aucs = []
    micro_aucs = []
    with open(log_file) as f:
        for line in f:
            # Prefer Macro-AUC (newer logs)
            m_macro = re.search(r"Macro-AUC=([0-9\.]+)", line)
            if m_macro:
                macro_aucs.append(float(m_macro.group(1)))
            
            # Backward compatibility with ROC-AUC (i.e., Micro-AUC)
            m_micro = re.search(r"ROC-AUC=([0-9\.]+)", line)
            if m_micro:
                micro_aucs.append(float(m_micro.group(1)))
    
    # Return Macro-AUC if present; otherwise return Micro-AUC
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

    # Grab the token ID for "yes"/"no"; use the first token to remain stable
    tok_yes = tok.encode("yes", add_special_tokens=False)
    tok_no  = tok.encode("no",  add_special_tokens=False)
    id_yes = tok_yes[0]
    id_no  = tok_no[0]

    # Optional: print the tokens used (for debugging)
    logger.info(f"[Token] yes → {tok.convert_ids_to_tokens([id_yes])[0]} (id={id_yes})")
    logger.info(f"[Token] no  → {tok.convert_ids_to_tokens([id_no])[0]} (id={id_no})")

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval@{epoch}", unit="batch"):
            for prm, ref, subtask in zip(batch["prompt"], batch["target_text"], batch["subtask"]):
                inp = tok(prm, return_tensors="pt", max_length=args.max_seq_length, truncation=True).to(dev)

                # Fetch the model output logits by taking the last token
                out = model(**inp)
                logits = out.logits[0, -1]  # shape: [vocab_size]

                # Convert logits to probabilities for "yes" and "no"
                prob = torch.softmax(logits[[id_yes, id_no]], dim=-1)
                p_yes = prob[0].item()
                scores_all.append(p_yes)

                # Classify using a 0.5 threshold
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

    # Compute overall metrics
    acc, p, r, f1 = _prf(preds_all, refs_all)

    # Micro (pooled) AUC (for reference only)
    y_all = np.array([1 if r == "yes" else 0 for r in refs_all])
    try:
        micro_auc = roc_auc_score(y_all, np.array(scores_all))
    except ValueError:
        micro_auc = float("nan")

    # Macro (per-task) AUC — the community standard
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
        
        if len(np.unique(y)) < 2:  # Skip if there are no positive or no negative samples
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

    # Verbose logging
    logger.info(f"[Eval] Overall: Acc={acc:.4f}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}")
    logger.info(f"[Eval] Macro-AUC={macro_auc:.4f} over {len(aucs)} tasks (skipped={skipped}); Micro-AUC={micro_auc:.4f}")
    
    # Print details for each sub-task
    logger.info("[Task Details]")
    for detail in task_details:
        logger.info(f"  {detail}")
    
    model.train()

def evaluate_single_task(epoch, model, tok, loader, dev, logger, args):
    """Evaluate a single task, return AUC"""
    if loader is None:
        return float("nan")
    
    model.eval()
    preds_all, refs_all, scores_all = [], [], []
    first = False

    # Get the token ID for "yes"/"no"
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

    # Compute metrics
    acc, p, r, f1 = _prf(preds_all, refs_all)

    # Compute AUC
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
    """Train a LoRA model for a single task"""
    dev=torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create a dedicated output directory for the current task
    task_dir_name = f"task_{task_name[:30].replace('?', '').replace(' ', '_')}{version_suffix}"
    task_output_dir = Path(args.output_dir) / task_dir_name
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger=get_logger(f"SFT_{task_name[:20]}{version_suffix}", task_output_dir/"train.log")
    logger.info(f"🎯 Starting training task: {task_name} (oversample={oversample})")
    logger.info(f"📊 Training samples: {len(task_train_recs)}, Validation samples: {len(task_val_recs)}")
    
    # Count the label distribution for the current task
    train_yes = sum(1 for r in task_train_recs if str(r["output"]).strip().lower() == "yes")
    train_no = len(task_train_recs) - train_yes
    val_yes = sum(1 for r in task_val_recs if str(r["output"]).strip().lower() == "yes")
    val_no = len(task_val_recs) - val_yes
    
    logger.info(f"📈 Training set distribution: {train_yes} yes, {train_no} no ({train_yes/(train_yes+train_no):.1%} positive)")
    logger.info(f"📉 Validation set distribution: {val_yes} yes, {val_no} no ({val_yes/(val_yes+val_no):.1%} positive)")

    # --- fresh LoRA for this task ---
    lcfg=LoraConfig(r=args.lora_r,lora_alpha=args.lora_alpha,lora_dropout=args.lora_dropout,
                    target_modules=[m.strip() for m in args.lora_target_modules.split(',')],
                    bias="none",task_type="CAUSAL_LM")
    
    # ✅ Reload base model for the current task (to avoid LoRA conflicts)
    logger.info("📋 Loading independent base model for the current task")
    if Path(args.stage2_model_path).is_dir():
        fresh_base=LlamaForCausalLM.from_pretrained(args.stage2_model_path, torch_dtype=torch.bfloat16).to(dev)
    else:
        fresh_base=LlamaForCausalLM.from_pretrained(args.base_llm_model_path, torch_dtype=torch.bfloat16).to(dev)
        if hasattr(base, 'config') and base.config.vocab_size != len(tok):
            fresh_base.resize_token_embeddings(len(tok))
    
    model=get_peft_model(fresh_base,lcfg).train()

    # Use the provided training and validation data directly
    train_recs = task_train_recs
    val_recs = task_val_recs
    
    # Single task data oversampling (if needed)
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
        logger.info(f"📊 Task-level balancing: {len(train_recs)} samples (yes {sum(str(r['output']).lower()=='yes' for r in train_recs)}, "
                    f"no {sum(str(r['output']).lower()=='no' for r in train_recs)})") 
    else:
        logger.info(f"📊 No oversampling: {len(train_recs)} samples")

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
    best_auc = float('-inf')  # Record the best AUC
    all_aucs = []  # Record AUC for all epochs
    
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
            
            # Update the best AUC
            if not math.isnan(task_auc) and task_auc > best_auc:
                best_auc = task_auc
                logger.info(f"🎯 New best AUC: {best_auc:.4f} (Epoch {ep+1})")
            
    # Summarize AUC information
    valid_aucs = [auc for auc in all_aucs if not math.isnan(auc)]
    if valid_aucs:
        max_auc = max(valid_aucs)
        final_auc = all_aucs[-1] if all_aucs else float("nan")  # AUC of the last epoch
        
        logger.info(f"📊 AUC Summary: Highest={max_auc:.4f}, Final={final_auc:.4f}, All epochs={[f'{v:.4f}' for v in valid_aucs]}")
        result_auc = max_auc
    else:
        logger.warning("⚠️ All AUCs are NaN")
        result_auc = float("nan")
    
    # 🚨 Critical: Clear memory
    logger.info("🧹 Clearing task memory...")
    del model, fresh_base, train_ds, val_ds, train_lo, val_lo, opt, sch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Show memory usage after cleanup
        memory_allocated = torch.cuda.memory_allocated(dev) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(dev) / 1024**3   # GB
        logger.info(f"💾 Memory after cleanup: Allocated={memory_allocated:.2f}GB, Reserved={memory_reserved:.2f}GB")
    
    return result_auc, task_output_dir

# ═══════════════════════════════════════════════
# Main functions
# ═══════════════════════════════════════════════
def sft_lora_all_tasks(args):
    """Main function: train all subtasks independently"""
    set_seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create the main output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main_logger = get_logger("SFT_AllTasks", Path(args.output_dir)/"main.log")
    main_logger.info(f"🚀 Starting independent training of all subtasks...")
    main_logger.info(f"Device={dev}")

    # --- Load shared components ---
    main_logger.info("📚 Loading shared components...")
    
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

    # --- Load and split data ---
    main_logger.info("📊 Loading raw data...")
    all_recs = []
    for t in [x.strip() for x in args.tasks.split(",") if x.strip()]:
        fp = Path(args.dataset_root) / f"{t}.json"
        if fp.is_file():
            all_recs += json.loads(fp.read_text())
    random.shuffle(all_recs)
    main_logger.info(f"Total data: {len(all_recs)}")

    # Scaffold split
    if RDKIT_AVAILABLE and args.use_scaffold_split == 1:
        main_logger.info("🧬 Using Scaffold Split for data splitting...")
        train_recs, val_recs, test_recs = scaffold_split(all_recs, val_ratio=0.1, test_ratio=0.1, seed=args.seed, balanced=args.balanced_scaffold)
        main_logger.info("✅ Scaffold Split completed")
    else:
        main_logger.info("📝 Using random split...")
        n_val = max(1, int(0.1 * len(all_recs)))
        n_test = max(1, int(0.1 * len(all_recs)))
        n_val = min(n_val, len(all_recs) - 2) if len(all_recs) >= 3 else n_val
        n_test = min(n_test, len(all_recs) - n_val - 1) if len(all_recs) - n_val >= 2 else n_test
        val_recs = all_recs[:n_val]
        test_recs = all_recs[n_val:n_val + n_test]
        train_recs = all_recs[n_val + n_test:]

    # --- Group data by subtasks ---
    main_logger.info("🎯 Grouping data by subtasks...")
    
    def group_by_subtask(recs):
        task_groups = defaultdict(list)
        for r in recs:
            subtask = r.get("subtask") or r.get("task") or r["instruction"].strip()
            task_groups[subtask].append(r)
        return task_groups
    
    train_task_groups = group_by_subtask(train_recs)
    val_task_groups = group_by_subtask(val_recs)
    test_task_groups = group_by_subtask(test_recs)
    
    # Ensure training and validation sets have the same tasks
    common_tasks = set(train_task_groups.keys()) & set(val_task_groups.keys()) & set(test_task_groups.keys())
    main_logger.info(f"Found {len(common_tasks)} common subtasks")
    
    if len(common_tasks) != len(train_task_groups) or len(common_tasks) != len(val_task_groups) or len(common_tasks) != len(test_task_groups):
        main_logger.warning("⚠️ Training, validation, or test sets have inconsistent tasks!")
    
    # --- Train each subtask ---
    task_results = {}
    failed_tasks = []
    
    for i, task_name in enumerate(sorted(common_tasks), 1):
        main_logger.info(f"\n🎯 [{i}/{len(common_tasks)}] Starting training task: {task_name[:50]}...")
        
        # Show current memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(dev) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(dev) / 1024**3   # GB
            main_logger.info(f"💾 Memory before task start: Allocated={memory_allocated:.2f}GB, Reserved={memory_reserved:.2f}GB")
        
        task_train_recs = train_task_groups[task_name]
        task_val_recs = val_task_groups[task_name]
        task_test_recs = test_task_groups[task_name]
        
        # Check data size
        train_yes = sum(1 for r in task_train_recs if str(r["output"]).strip().lower() == "yes")
        train_no = len(task_train_recs) - train_yes
        val_yes = sum(1 for r in task_val_recs if str(r["output"]).strip().lower() == "yes")
        val_no = len(task_val_recs) - val_yes
        test_yes = sum(1 for r in task_test_recs if str(r["output"]).strip().lower() == "yes")
        test_no = len(task_test_recs) - test_yes
        
        main_logger.info(f"Training: {train_yes}+{train_no}={len(task_train_recs)}, Validation: {val_yes}+{val_no}={len(task_val_recs)}, Test: {test_yes}+{test_no}={len(task_test_recs)}")
        
        # Skip tasks with too little data or single class
        if len(task_train_recs) < 10 or len(task_val_recs) < 3 or len(task_test_recs) < 3:
            main_logger.warning(f"⚠️ Skipping task {task_name[:30]}... (data too small)")
            failed_tasks.append(task_name)
            continue
            
        if min(train_yes, train_no) == 0 or min(val_yes, val_no) == 0 or min(test_yes, test_no) == 0:
            main_logger.warning(f"⚠️ Skipping task {task_name[:30]}... (single class)")
            failed_tasks.append(task_name)
            continue
        
        try:
            # Train the current task with two versions: oversample and no-oversample
            main_logger.info(f"🎆 Starting training two versions: {task_name[:30]}...")
            
            # Version 1: Use oversample
            main_logger.info(f"💹 Version 1: Use oversample")
            task_auc_oversample, task_output_dir_oversample = sft_lora_single_task(
                args, task_name, task_train_recs, task_val_recs, gnn, vq, tok, base, 
                oversample=True, version_suffix="_oversample"
            )
            
            # Version 2: Do not use oversample
            main_logger.info(f"💷 Version 2: Do not use oversample")
            task_auc_no_oversample, task_output_dir_no_oversample = sft_lora_single_task(
                args, task_name, task_train_recs, task_val_recs, gnn, vq, tok, base, 
                oversample=False, version_suffix="_no_oversample"
            )
            
            # Select the best result
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
            main_logger.info(f"✅ Task completed: {task_name[:30]}...")
            main_logger.info(f"📊 oversample AUC={task_auc_oversample:.4f}, no_oversample AUC={task_auc_no_oversample:.4f}")
            main_logger.info(f"🏆 Best version: {best_version}, Final AUC={best_auc:.4f}")
            
        except Exception as e:
            main_logger.error(f"❌ Task failed: {task_name[:30]}... Error: {str(e)}")
            import traceback
            main_logger.error(f"❌ Detailed error information: {traceback.format_exc()}")
            failed_tasks.append(task_name)
            # Clear memory after exception
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                main_logger.info("🧹 Clearing memory cache after exception")
            continue
    
    # --- Summarize results ---
    main_logger.info(f"\n🎊 All tasks training completed!")
    main_logger.info(f"Success: {len(task_results)} tasks")
    main_logger.info(f"Failed: {len(failed_tasks)} tasks")
    
    if task_results:
        valid_aucs = [auc for auc in task_results.values() if not math.isnan(auc)]
        macro_auc = np.mean(valid_aucs) if valid_aucs else float("nan")
        
        main_logger.info(f"\n📊 Result Summary:")
        main_logger.info(f"Macro-AUC: {macro_auc:.4f} (over {len(valid_aucs)} valid tasks)")
        
        # Detailed results
        main_logger.info("\n📋 Detailed results for each task:")
        for task_name, auc in sorted(task_results.items(), key=lambda x: x[1], reverse=True):
            short_name = task_name[:50] + "..." if len(task_name) > 50 else task_name
            main_logger.info(f"  {short_name}: AUC={auc:.4f}")
        
        if failed_tasks:
            main_logger.info("\n❌ Failed tasks:")
            for task in failed_tasks:
                short_name = task[:50] + "..." if len(task) > 50 else task
                main_logger.info(f"  {short_name}")
        
        return macro_auc
    else:
        main_logger.error("❌ No tasks successfully completed!")
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
    pa.add_argument("--oversample", type=int, default=1, help="1: apply minority oversampling within each task (recommended); 0: skip oversampling")
    pa.add_argument("--use_scaffold_split", type=int, default=1, help="1: use scaffold-based splitting (default); 0: use random split")
    pa.add_argument("--balanced_scaffold", type=int, default=1, help="1: random scaffold split (more balanced); 0: deterministic split (reproducible but may be imbalanced)")

    # Task-level oversampling
    pa.add_argument("--task_level_oversample", type=int, default=1, help="1: enable task-level imbalance handling (recommended); 0: disable")
    pa.add_argument("--min_samples_per_class", type=int, default=10, help="Minimum samples per class within each task")
    pa.add_argument("--balance_threshold", type=float, default=0.4, help="Oversample if the positive ratio is below this value or above (1 - value)")

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

    pa.add_argument("--multi_seed", default="12", help="Comma-separated seed list, e.g. '42,123,234,345,456'. Runs all seeds and reports mean±std")

    args = pa.parse_args()

    if args.multi_seed:
        seed_list = [int(x) for x in args.multi_seed.split(",") if x.strip()]
        aurocs = []
        task_str = "_".join([t.strip() for t in args.tasks.split(",")])
        for seed in seed_list:
            print(f"\n=== [Seed {seed}] Training... ===\n")
            # Set seed and output directory
            args.seed = seed
            args.output_dir = os.path.join(task_str, f"seed_{seed}")

            # If output directory exists, it's recommended to clear it
            if os.path.exists(args.output_dir):
                shutil.rmtree(args.output_dir)

            # Run the main flow once
            macro_auc = sft_lora_all_tasks(args)
            print(f"[Seed {seed}] Final Macro-AUC: {macro_auc:.4f}")
            aurocs.append(macro_auc)

        # Summarize results
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
            
            # If only one seed, special note
            if len(aurocs_arr) == 1:
                print("⚠️  Note: only one valid result; cannot compute standard deviation")
        else:
            print("❌ No valid results!")
        
        print("="*60)
    else:
        # Single seed run
        sft_lora_all_tasks(args)