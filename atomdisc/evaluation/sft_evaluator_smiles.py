# sft_evaluator_smiles.py  (real-time sample logging)
"""
Evaluator & logging utilities with on-the-fly sample output
-------------------------------------------------------------
- After processing each batch, log every sample's Prompt / Reference / Prediction
  along with Exact Match, RDK/MACCS/Morgan similarities.
- At the end of evaluation, still save the full metrics and all sample outputs.
"""

import os
import logging
from typing import List

from tqdm import tqdm
import torch

from compute_metrics_smiles import (
    compute_exact_match,
    compute_bleu_on_smiles,
    compute_avg_levenshtein,
    compute_fingerprint_similarities,
)

# -----------------------------------------------------------------------------
# Evaluation main function
# -----------------------------------------------------------------------------
def evaluate_reaction_prediction(
    current_epoch_num: int,
    model_to_eval,
    eval_tokenizer,
    eval_dataloader,
    eval_device,
    main_script_logger: logging.Logger,
    sft_save_dir: str,
    script_args,
):
    """Evaluate reaction prediction / retrosynthesis tasks and record metrics."""
    from compute_metrics_smiles import (
        compute_exact_match,
        compute_bleu_on_smiles,
        compute_avg_levenshtein,
        compute_fingerprint_similarities
    )
    import Levenshtein

    main_script_logger.info(f"--- 🧪 Start Evaluation (Epoch {current_epoch_num}) ---")
    model_to_eval.eval()

    all_predictions, all_references, all_prompts = [], [], []

    for batch in tqdm(eval_dataloader, desc="Evaluating", unit="batch"):
        prompts = batch["prompt"]
        references = batch["target_text"]
        batch_predictions = []

        for prompt_text in prompts:
            inputs = eval_tokenizer(prompt_text, return_tensors="pt", max_length=script_args.max_seq_length, truncation=True).to(eval_device)
            gen_ids = model_to_eval.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=script_args.max_new_tokens_eval,
                num_beams=script_args.num_beams_eval,
                do_sample=script_args.do_sample_eval,
                pad_token_id=eval_tokenizer.pad_token_id,
                eos_token_id=eval_tokenizer.eos_token_id,
            )
            prompt_len = inputs['input_ids'].shape[1]
            decoded = eval_tokenizer.decode(gen_ids[0, prompt_len:], skip_special_tokens=True)
            batch_predictions.append(decoded.strip())

        all_predictions.extend(batch_predictions)
        all_references.extend([r.strip() for r in references])
        all_prompts.extend(prompts)

        # Log each sample and metrics in real time
        for i, (p_text, r_text, prompt_text) in enumerate(zip(batch_predictions, references, prompts)):
            em = int(p_text.strip() == r_text.strip())
            bleu_one = compute_bleu_on_smiles([p_text.strip()], [r_text.strip()])
            lev_dist = compute_avg_levenshtein([p_text.strip()], [r_text.strip()])
            rdk, maccs, morgan = compute_fingerprint_similarities([p_text], [r_text])[3][0]
            main_script_logger.info(
                f"\n--- 🔍 Eval Sample ---\n"
                f"Prompt:\n{prompt_text}\n"
                f"Reference:\n{r_text.strip()}\n"
                f"Prediction:\n{p_text.strip()}\n"
                f"Exact Match: {em} | BLEU: {bleu_one:.3f} | Levenshtein: {lev_dist} | RDK: {rdk:.3f} | MACCS: {maccs:.3f} | Morgan: {morgan:.3f}\n"
            )

    if not all_predictions:
        main_script_logger.warning("Empty predictions; skip metric computation.")
        return {
            "Exact Match ↑": 0,
            "BLEU ↑": 0,
            "Levenshtein ↓": float("inf"),
            "RDK FTS ↑": 0,
            "MACCS FTS ↑": 0,
            "Morgan FTS ↑": 0,
        }

    exact_match = compute_exact_match(all_predictions, all_references)
    bleu = compute_bleu_on_smiles(all_predictions, all_references)
    levenshtein = compute_avg_levenshtein(all_predictions, all_references)
    _, _, _, fps_scores = compute_fingerprint_similarities(all_predictions, all_references)
    avg_rdk = sum(score[0] for score in fps_scores) / len(fps_scores)
    avg_maccs = sum(score[1] for score in fps_scores) / len(fps_scores)
    avg_morgan = sum(score[2] for score in fps_scores) / len(fps_scores)

    metrics = {
        "epoch": current_epoch_num,
        "Exact Match ↑": f"{exact_match:.4f}",
        "BLEU ↑":        f"{bleu:.4f}",
        "Levenshtein ↓": f"{levenshtein:.4f}",
        "RDK FTS ↑":     f"{avg_rdk:.4f}",
        "MACCS FTS ↑":   f"{avg_maccs:.4f}",
        "Morgan FTS ↑":  f"{avg_morgan:.4f}",
    }

    os.makedirs(sft_save_dir, exist_ok=True)
    out_path = os.path.join(sft_save_dir, f"epoch_{current_epoch_num}_eval_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Metrics: {metrics}\n\n")
        for i, (p, r, prompt) in enumerate(zip(all_predictions, all_references, all_prompts)):
            em = int(p.strip() == r.strip())
            bleu_one = compute_bleu_on_smiles([p.strip()], [r.strip()])
            lev_dist = Levenshtein.distance(p.strip(), r.strip())
            rdk, maccs, morgan = compute_fingerprint_similarities([p], [r])[3][0]
            f.write(f"--- Example {i+1} ---\n")
            f.write(f"Prompt:\n{prompt}\n")
            f.write(f"Reference (Target):\n{r}\n")
            f.write(f"Prediction (Generated):\n{p}\n")
            f.write(f"Exact Match: {em} | BLEU: {bleu_one:.3f} | Levenshtein: {lev_dist} | RDK: {rdk:.3f} | MACCS: {maccs:.3f} | Morgan: {morgan:.3f}\n\n")

    main_script_logger.info(f"Metrics & all samples saved → {out_path}")
    model_to_eval.train()
    return metrics

def evaluate_retrosynthesis_prediction(
    current_epoch_num: int,
    model_to_eval,
    eval_tokenizer,
    eval_dataloader,
    eval_device,
    main_script_logger: logging.Logger,
    sft_save_dir: str,
    script_args,
):
    """Evaluate reaction prediction / retrosynthesis tasks and record metrics."""
    from compute_metrics_smiles import (
        compute_exact_match,
        compute_bleu_on_smiles,
        compute_avg_levenshtein,
        compute_fingerprint_similarities
    )
    import Levenshtein

    main_script_logger.info(f"--- 🧪 Start Evaluation (Epoch {current_epoch_num}) ---")
    model_to_eval.eval()

    all_predictions, all_references, all_prompts = [], [], []

    for batch in tqdm(eval_dataloader, desc="Evaluating", unit="batch"):
        prompts = batch["prompt"]
        references = batch["target_text"]
        batch_predictions = []

        for prompt_text in prompts:
            inputs = eval_tokenizer(prompt_text, return_tensors="pt", max_length=script_args.max_seq_length, truncation=True).to(eval_device)
            gen_ids = model_to_eval.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=script_args.max_new_tokens_eval,
                num_beams=script_args.num_beams_eval,
                do_sample=script_args.do_sample_eval,
                pad_token_id=eval_tokenizer.pad_token_id,
                eos_token_id=eval_tokenizer.eos_token_id,
            )
            def join_smiles_lines(s):
                """Collapse multi-line SMILES into a single line (join with '.')."""
                return ".".join([i.strip() for i in s.strip().splitlines() if i.strip()])

            prompt_len = inputs['input_ids'].shape[1]
            decoded = eval_tokenizer.decode(gen_ids[0, prompt_len:], skip_special_tokens=True)
            batch_predictions.append(join_smiles_lines(decoded))

        all_predictions.extend(batch_predictions)
        all_references.extend([r.strip() for r in references])
        all_prompts.extend(prompts)

        # Log each sample and metrics in real time
        for i, (p_text, r_text, prompt_text) in enumerate(zip(batch_predictions, references, prompts)):
            em = int(p_text.strip() == r_text.strip())
            bleu_one = compute_bleu_on_smiles([p_text.strip()], [r_text.strip()])
            lev_dist = compute_avg_levenshtein([p_text.strip()], [r_text.strip()])
            rdk, maccs, morgan = compute_fingerprint_similarities([p_text], [r_text])[3][0]
            main_script_logger.info(
                f"\n--- 🔍 Eval Sample ---\n"
                f"Prompt:\n{prompt_text}\n"
                f"Reference:\n{r_text.strip()}\n"
                f"Prediction:\n{p_text.strip()}\n"
                f"Exact Match: {em} | BLEU: {bleu_one:.3f} | Levenshtein: {lev_dist} | RDK: {rdk:.3f} | MACCS: {maccs:.3f} | Morgan: {morgan:.3f}\n"
            )

    if not all_predictions:
        main_script_logger.warning("Empty predictions; skip metric computation.")
        return {
            "Exact Match ↑": 0,
            "BLEU ↑": 0,
            "Levenshtein ↓": float("inf"),
            "RDK FTS ↑": 0,
            "MACCS FTS ↑": 0,
            "Morgan FTS ↑": 0,
        }

    exact_match = compute_exact_match(all_predictions, all_references)
    bleu = compute_bleu_on_smiles(all_predictions, all_references)
    levenshtein = compute_avg_levenshtein(all_predictions, all_references)
    _, _, _, fps_scores = compute_fingerprint_similarities(all_predictions, all_references)
    avg_rdk = sum(score[0] for score in fps_scores) / len(fps_scores)
    avg_maccs = sum(score[1] for score in fps_scores) / len(fps_scores)
    avg_morgan = sum(score[2] for score in fps_scores) / len(fps_scores)

    metrics = {
        "epoch": current_epoch_num,
        "Exact Match ↑": f"{exact_match:.4f}",
        "BLEU ↑":        f"{bleu:.4f}",
        "Levenshtein ↓": f"{levenshtein:.4f}",
        "RDK FTS ↑":     f"{avg_rdk:.4f}",
        "MACCS FTS ↑":   f"{avg_maccs:.4f}",
        "Morgan FTS ↑":  f"{avg_morgan:.4f}",
    }

    os.makedirs(sft_save_dir, exist_ok=True)
    out_path = os.path.join(sft_save_dir, f"epoch_{current_epoch_num}_eval_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Metrics: {metrics}\n\n")
        for i, (p, r, prompt) in enumerate(zip(all_predictions, all_references, all_prompts)):
            em = int(p.strip() == r.strip())
            bleu_one = compute_bleu_on_smiles([p.strip()], [r.strip()])
            lev_dist = Levenshtein.distance(p.strip(), r.strip())
            rdk, maccs, morgan = compute_fingerprint_similarities([p], [r])[3][0]
            f.write(f"--- Example {i+1} ---\n")
            f.write(f"Prompt:\n{prompt}\n")
            f.write(f"Reference (Target):\n{r}\n")
            f.write(f"Prediction (Generated):\n{p}\n")
            f.write(f"Exact Match: {em} | BLEU: {bleu_one:.3f} | Levenshtein: {lev_dist} | RDK: {rdk:.3f} | MACCS: {maccs:.3f} | Morgan: {morgan:.3f}\n\n")

    main_script_logger.info(f"Metrics & all samples saved → {out_path}")
    model_to_eval.train()
    return metrics

# -----------------------------------------------------------------------------
# Simple training-time sample logger (unchanged)
# -----------------------------------------------------------------------------

def log_sample_generation(model, tokenizer, batch_data, device, logger, args):
    model.eval()
    prompt_text    = batch_data["prompt"][0]
    reference_text = batch_data["target_text"][0]

    inputs = tokenizer(prompt_text, return_tensors="pt", max_length=args.max_seq_length, truncation=True).to(device)
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens_log,
            num_beams=args.num_beams_log,
            do_sample=args.do_sample_log,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    pred = tokenizer.decode(gen_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    logger.info(
        f"\n--- 💡 Training Sample Log ---\n"
        f"Prompt:\n{prompt_text}\n"
        f"Reference:\n{reference_text}\n"
        f"Prediction:\n{pred}\n"
        f"---------------------------------\n"
    )
    model.train()