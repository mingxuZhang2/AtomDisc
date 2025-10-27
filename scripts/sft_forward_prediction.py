# stage4_sft_lora.py
import os
import argparse
import logging
import math
import functools
import json
import random

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, LlamaForCausalLM, get_linear_schedule_with_warmup
from peft import PeftModel, LoraConfig, get_peft_model

from atomdisc.utils.gnn_vq_utils import (
    set_seed,
    get_logger,
    load_gnn_vq_models,
)
from atomdisc.tokenization.mol_tokenizer import convert_text_smiles_to_mol_tokens

from atomdisc.evaluation.sft_evaluator_smiles import (
    evaluate_reaction_prediction,
    log_sample_generation,
)

from atomdisc.datasets.forward_dataset import MolVQDataset, collate_fn as user_collate_fn

def log_sample_generation_test(model, tokenizer, batch_data, device, logger, args):
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

def sft_lora(args):
    # -------------------- Setup --------------------
    set_seed(args.seed)
    
    device_str = args.device.lower()
    if device_str == "cpu":
        effective_device = torch.device("cpu")
    elif device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            if not logging.getLogger("Stage4_SFT_LoRA").hasHandlers():
                 logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])
            logging.warning(f"CUDA specified ({args.device}) but not available. Defaulting to CPU.")
            effective_device = torch.device("cpu")
        elif ":" not in device_str: 
             effective_device = torch.device("cuda:0")
        else: 
             effective_device = torch.device(device_str)
    elif device_str.isdigit(): 
        if not torch.cuda.is_available():
            if not logging.getLogger("Stage4_SFT_LoRA").hasHandlers():
                 logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])
            logging.warning(f"CUDA specified by index ({args.device}) but not available. Defaulting to CPU.")
            effective_device = torch.device("cpu")
        else:
            effective_device = torch.device(f"cuda:{device_str}")
    else:
        if not logging.getLogger("Stage4_SFT_LoRA").hasHandlers():
             logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])
        logging.warning(f"Invalid device string '{args.device}'. Defaulting to CPU.")
        effective_device = torch.device("cpu")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        
    logger = get_logger(
        "Stage4_SFT_LoRA",
        os.path.join(args.output_dir, "stage4_sft_lora.log")
    )
    logger.info(f"Effective device selected: {effective_device}")
    logger.info(f"Script arguments: {args}")

    if torch.cuda.is_available() and effective_device.type == 'cuda':
        logger.info("Clearing CUDA cache at the start...")
        torch.cuda.empty_cache()

    # -------------------- Load Frozen GNN and VQ from Stage 1 --------------------
    logger.info(f"Loading frozen GNN and VQ from: {args.stage1_gnn_vq_checkpoint_path}")
    if not os.path.exists(args.stage1_gnn_vq_checkpoint_path):
        logger.error(f"Stage 1 GNN/VQ checkpoint not found at {args.stage1_gnn_vq_checkpoint_path}.")
        return
    try:
        gnn, vq = load_gnn_vq_models(args.stage1_gnn_vq_checkpoint_path, effective_device)
        logger.info("Frozen GNN and VQ models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load GNN/VQ models from Stage 1 checkpoint: {e}", exc_info=True)
        return

    # -------------------- Load or Initialize LLM and Tokenizer --------------------
    if args.stage2_model_path and os.path.exists(args.stage2_model_path):
        logger.info(f"Loading LLM with trained embeddings and tokenizer from Stage 2 output: {args.stage2_model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.stage2_model_path, use_fast=True)
            llm_base_model = LlamaForCausalLM.from_pretrained(
                args.stage2_model_path, 
                torch_dtype=torch.bfloat16, 
            ).to(effective_device)
            logger.info(f"Base LLM (from Stage 2) loaded. Vocab size: {len(tokenizer)}")
        except Exception as e:
            logger.error(f"Failed to load LLM/tokenizer from Stage 2: {e}", exc_info=True)
            return
    else:
        logger.warning("No valid Stage 2 model path provided. Initializing from base LLM and adding special tokens.")
        try:
            # Load base tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer_path, use_fast=True)
            llm_base_model = LlamaForCausalLM.from_pretrained(
                args.base_llm_model_path,
                torch_dtype=torch.bfloat16,
            ).to(effective_device)
            
            # Define and add special tokens
            if args.use_structure_token:
                special_tokens_to_add = ['<mol>', '</mol>'] + [f'<atom_{i}>' for i in range(args.vq_codebook_size)]
                tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})
                
                # Resize model embeddings
                llm_base_model.resize_token_embeddings(len(tokenizer))
                logger.info(f"Loaded base LLM and tokenizer. Added {len(special_tokens_to_add)} special tokens.")
                logger.info(f"New vocab size: {len(tokenizer)}. New token embeddings are randomly initialized.")
            else:
                logger.info(f"No Special Tokens added. Current vocab size: {len(tokenizer)}.")
        except Exception as e:
            logger.error(f"Failed to load base LLM/tokenizer and add special tokens: {e}", exc_info=True)
            return

    # Set pad token if it's not already set
    if tokenizer.pad_token is None:
        logger.info("Tokenizer pad_token not set. Setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # -------------------- Load or Initialize LoRA Adapters --------------------
    if args.stage3_lora_adapter_path and os.path.exists(args.stage3_lora_adapter_path):
        logger.info(f"Loading pre-trained LoRA adapters from: {args.stage3_lora_adapter_path}")
        try:
            peft_model_sft = PeftModel.from_pretrained(
                llm_base_model, 
                args.stage3_lora_adapter_path,
                is_trainable=True  
            )
            logger.info("Successfully loaded pre-trained LoRA adapters and set to trainable.")
        except Exception as e:
            logger.error(f"Failed to load LoRA adapters from Stage 3 path: {e}", exc_info=True)
            return
    else:
        logger.info("No valid Stage 3 LoRA adapter path provided. Initializing a new LoRA model for SFT.")
        lora_target_modules = [m.strip() for m in args.lora_target_modules.split(',')]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model_sft = get_peft_model(llm_base_model, lora_config)
        logger.info("New LoRA model initialized and applied to the base LLM.")
    
    peft_model_sft.train() 
    logger.info("PEFT model (LLM + LoRA) set to train mode for SFT.")
    peft_model_sft.print_trainable_parameters() 

    if args.train_special_embeddings_in_sft:
        logger.info("Special token embeddings will ALSO be fine-tuned alongside LoRA layers in SFT.")
        input_embeddings = peft_model_sft.get_input_embeddings() 
        
        if args.original_base_llm_vocab_size is None:
            logger.error("--original_base_llm_vocab_size is required if --train_special_embeddings_in_sft is set.")
            return

        trainable_embedding_mask = torch.zeros_like(input_embeddings.weight.data, dtype=torch.bool) 
        trainable_embedding_mask[args.original_base_llm_vocab_size:, :] = True 
        
        input_embeddings.weight.requires_grad = True 
        
        if hasattr(input_embeddings.weight, '_backward_hooks') and input_embeddings.weight._backward_hooks:
             logger.warning("Clearing existing backward hooks on embedding weight.")
             input_embeddings.weight._backward_hooks.clear()

        def embedding_grad_hook_sft(grad):
            return grad * trainable_embedding_mask.to(grad.device) 
        input_embeddings.weight.register_hook(embedding_grad_hook_sft)
        logger.info(f"Hook registered to fine-tune special embeddings (indices {args.original_base_llm_vocab_size} to {len(tokenizer)-1}).")
        peft_model_sft.print_trainable_parameters() 
    else:
        logger.info("Special token embeddings will be FROZEN during SFT (LoRA layers are trainable).")


    # -------------------- Dataset and DataLoader for SFT (Using User's MolVQDataset) --------------------
    logger.info(f"Initializing SFT Dataset using MolVQDataset from: {args.sft_data_path}")
    try:
        with open(args.sft_data_path, 'r', encoding='utf-8') as f:
            sft_records_raw = json.load(f) # Assuming the SFT data is a list of records in a JSON file
        logger.info(f"   Loaded {len(sft_records_raw)} raw records for SFT.")
    except Exception as e:
        logger.error(f"Failed to load SFT data from {args.sft_data_path}: {e}", exc_info=True)
        return

    random.shuffle(sft_records_raw) # Shuffle before splitting
    train_sft_records = [rec for rec in sft_records_raw if rec.get('metadata', {}).get('split') == 'train']
    val_sft_records   = [rec for rec in sft_records_raw if rec.get('metadata', {}).get('split') == 'test']
    logger.info(f"SFT Dataset split: {len(train_sft_records)} training records, {len(val_sft_records)} validation/test records.")

    # Split SFT data
    train_size = int(args.sft_train_split_ratio * len(train_sft_records))
    val_size = len(sft_records_raw) - train_size
    if val_size == 0 and len(sft_records_raw) > 0 : 
        if train_size > 1 :
            train_size -=1
            val_size +=1
        else: 
            logger.warning("SFT Dataset too small to create a validation set. Using all data for training and validation.")

    logger.info(f"SFT Dataset split: {len(train_sft_records)} training records, {len(val_sft_records)} validation records.")

    # Instantiate user's MolVQDataset
    train_sft_dataset = MolVQDataset(
        train_sft_records, tokenizer, gnn, vq, effective_device, 
        max_length_prompt=args.max_prompt_len, 
        max_length_response=args.max_response_len,
        use_structure_token=args.use_structure_token
    )
    if len(val_sft_records) > 0:
        val_sft_dataset = MolVQDataset(
            val_sft_records, tokenizer, gnn, vq, effective_device,
            max_length_prompt=args.max_prompt_len,
            max_length_response=args.max_response_len,
            use_structure_token=args.use_structure_token
        )
    else:
        val_sft_dataset = None # No validation data

    # Prepare collate_fn with pad_token_id
    custom_collate_for_dataloader = functools.partial(user_collate_fn, pad_token_id=tokenizer.pad_token_id)
    
    train_dataloader = DataLoader( 
        train_sft_dataset,
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        collate_fn=custom_collate_for_dataloader, 
        drop_last=True 
    )
    if val_sft_dataset:
        val_dataloader = DataLoader(
            val_sft_dataset,
            batch_size=args.eval_batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=custom_collate_for_dataloader,
            drop_last=False
        )
        logger.info(f"SFT DataLoaders created. Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}.")
    else:
        val_dataloader = None
        logger.info(f"SFT Train DataLoader created. Train batches: {len(train_dataloader)}. No validation DataLoader.")


    # -------------------- Optimizer and Scheduler --------------------
    trainable_params = [p for p in peft_model_sft.parameters() if p.requires_grad]
    logger.info(f"Number of trainable parameters for SFT: {sum(p.numel() for p in trainable_params)}")
    if not trainable_params:
        logger.error("No trainable parameters found for SFT! Check LoRA setup and --train_special_embeddings_in_sft flag.")
        return

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    num_optimizer_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    total_optimizer_steps = num_optimizer_steps_per_epoch * args.num_epochs
    
    scheduler = get_linear_schedule_with_warmup( 
        optimizer, 
        num_warmup_steps=int(args.warmup_ratio * total_optimizer_steps) if total_optimizer_steps > 0 else 0, 
        num_training_steps=total_optimizer_steps if total_optimizer_steps > 0 else 100
    )

    logger.info(f"Optimizer and Scheduler configured for SFT. Total optimizer steps: {total_optimizer_steps}")

    # -------------------- SFT Training Loop --------------------
    logger.info(f"🚀 Starting SFT (LoRA fine-tuning) for {args.num_epochs} epochs...")
    global_step = 0 
    optimizer_step_count = 0 

    logger.info(f"🚀 Initial Evaluation")
    eval_metrics = evaluate_reaction_prediction(
                current_epoch_num=0, model_to_eval=peft_model_sft,
                eval_tokenizer=tokenizer, eval_dataloader=val_dataloader, 
                eval_device=effective_device, main_script_logger=logger,
                sft_save_dir=args.output_dir, script_args=args
            )
    logger.info(f"SFT Epoch 0 Validation Metrics: {str(eval_metrics)}")

    for epoch in range(args.num_epochs):
        peft_model_sft.train() 
        epoch_ce_loss_sum = 0
        optimizer.zero_grad() 

        train_iterator = tqdm(train_dataloader, desc=f"SFT Epoch {epoch+1}/{args.num_epochs}", unit="batch")
        
        for batch_idx, batch_data in enumerate(train_iterator):
            input_ids = batch_data['input_ids'].to(effective_device)
            attention_mask = batch_data['attention_mask'].to(effective_device)
            labels = batch_data['labels'].to(effective_device)
            
            log_batch_for_sample = {
                "input_ids": input_ids, "attention_mask": attention_mask,
                "prompt": batch_data["prompt"], "target_text": batch_data["target_text"],
                "response": batch_data["response"] 
            }

            outputs = peft_model_sft(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            ce_loss = outputs.loss 
            
            if torch.isnan(ce_loss) or torch.isinf(ce_loss):
                logger.warning(f"NaN or Inf CE loss detected at SFT Epoch {epoch+1}, Batch {batch_idx+1}. Skipping.")
                del input_ids, attention_mask, labels, outputs, ce_loss, batch_data
                if torch.cuda.is_available() and effective_device.type == 'cuda': torch.cuda.empty_cache()
                continue

            ce_loss_normalized = ce_loss / args.gradient_accumulation_steps
            ce_loss_normalized.backward() 
            epoch_ce_loss_sum += ce_loss.item() 
            global_step += 1
            train_iterator.set_postfix({"loss": f"{ce_loss.item():.4f}", "OptSteps": optimizer_step_count})

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad() 
                optimizer_step_count += 1

            if global_step % args.logging_steps == 0: 
                avg_ce_loss_epoch = epoch_ce_loss_sum / (batch_idx + 1) if (batch_idx + 1) > 0 else 0
                current_lr = scheduler.get_last_lr()[0] 
                logger.info(f"SFT Epoch {epoch+1}, MicroStep {global_step}, OptStep {optimizer_step_count}, LR: {current_lr:.2e}, AvgEpochCE: {avg_ce_loss_epoch:.4f}")

            if global_step % args.sample_output_steps == 0: 
                log_sample_generation(
                    peft_model_sft,
                    tokenizer, log_batch_for_sample, 
                    effective_device, logger, args
                )
            del input_ids, attention_mask, labels, outputs, ce_loss, ce_loss_normalized, batch_data, log_batch_for_sample

        if (len(train_dataloader) % args.gradient_accumulation_steps != 0):
            logger.info("Performing final optimizer step for SFT epoch.")
            if args.max_grad_norm > 0:
                 torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            optimizer_step_count +=1 

        avg_epoch_ce_loss = epoch_ce_loss_sum / len(train_dataloader) if len(train_dataloader) > 0 else 0
        logger.info(f"SFT Epoch {epoch+1} completed. Optimizer Steps: {optimizer_step_count}. Average CE Loss: {avg_epoch_ce_loss:.4f}")
            
        if val_dataloader and (epoch + 1) % args.eval_every_epochs == 0 :
            eval_metrics = evaluate_reaction_prediction(
                current_epoch_num=epoch + 1, model_to_eval=peft_model_sft,
                eval_tokenizer=tokenizer, eval_dataloader=val_dataloader, 
                eval_device=effective_device, main_script_logger=logger,
                sft_save_dir=args.output_dir, script_args=args
            )
            logger.info(f"SFT Epoch {epoch+1} Validation Metrics: {str(eval_metrics)}")
        else:
            logger.info(f"SFT Epoch {epoch+1}: Skipping validation (no validation data or not an eval epoch).")

        if torch.cuda.is_available() and effective_device.type == 'cuda':
            logger.info(f"Clearing CUDA cache at the end of SFT epoch {epoch+1}...")
            torch.cuda.empty_cache()

        if (epoch + 1) % args.save_every_epochs == 0 or (epoch + 1) == args.num_epochs:
            current_output_adapter_dir = args.output_dir
            if args.save_intermediate_checkpoints and (epoch + 1) != args.num_epochs:
                 current_output_adapter_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}_sft_lora_adapters")
            else: 
                 current_output_adapter_dir = os.path.join(args.output_dir, "final_sft_lora_adapters")

            if not os.path.exists(current_output_adapter_dir):
                os.makedirs(current_output_adapter_dir, exist_ok=True)
            
            logger.info(f"Saving SFT LoRA adapters to {current_output_adapter_dir}")
            peft_model_sft.save_pretrained(current_output_adapter_dir) 
            tokenizer.save_pretrained(current_output_adapter_dir) 
            logger.info(f"SFT LoRA adapters and Tokenizer saved for epoch {epoch+1} to {current_output_adapter_dir}")

    logger.info("Stage 4 (SFT LoRA) Fine-tuning finished.")
    final_save_path = os.path.join(args.output_dir, "final_sft_lora_adapters")
    logger.info(f"Final SFT LoRA adapters should be in {final_save_path} (if num_epochs was reached).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Supervised Fine-Tuning (SFT) of LoRA adapters.")
    
    # Paths
    parser.add_argument("--sft_data_path", type=str, default="", help="Path to the JSON file for SFT.")
    parser.add_argument("--stage1_gnn_vq_checkpoint_path", type=str, default="", help="Path to the Stage-1 GNN/VQ checkpoint.")
    parser.add_argument("--use_structure_token", type=int, default=1, help="1: use special token; 0: do not use")
    # --- NEW: Paths for initializing from a base LLM ---
    
    parser.add_argument("--stage2_model_path", type=str, default="", help="Path to the Stage-2 model (optional).")
    '''
    parser.add_argument("--stage2_model_path", type=str, 
                        default="", 
                        help="Path to the Stage 2 LLM with pre-trained embeddings. Leave empty to initialize from a base model.")
    '''
    parser.add_argument("--base_tokenizer_path", type=str, default="", help="Base tokenizer directory (used if Stage-2 path empty).")
    parser.add_argument("--base_llm_model_path", type=str, default="", help="Base LLM directory (used if Stage-2 path empty).")
    # --- END NEW ---
    '''                 
    parser.add_argument("--stage3_lora_adapter_path", type=str, 
                        default="", help="Path to pre-trained Stage-3 LoRA adapters (optional).")
    '''   
    parser.add_argument("--stage3_lora_adapter_path", type=str, default="", help="Stage-3 LoRA adapter path (optional).")
    parser.add_argument("--output_dir", type=str, default="./stage4_sft_forward", help="Directory to save outputs.")
    
    # --- NEW: VQ codebook size for adding special tokens ---
    parser.add_argument("--vq_codebook_size", type=int, default=512,
                        help="Size of the VQ codebook, needed to add the correct number of <atom_k> tokens if starting from a base LLM.")
    # --- END NEW ---

    parser.add_argument("--original_base_llm_vocab_size", type=int, required=False, 
                        help="Original vocabulary size of the base LLM. Required only if --train_special_embeddings_in_sft is set.")

    # Training hparams
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of SFT epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro-batch size for SFT.")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=6, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for SFT.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--sft_train_split_ratio", type=float, default=0.98, help="Train/validation split ratio.")

    # LoRA hparams
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA r (rank).") 
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj,k_proj,o_proj", 
                        help="Comma-separated list of LoRA target modules.")
    parser.add_argument("--train_special_embeddings_in_sft", action='store_true',
                        help="If set, fine-tune special token embeddings alongside LoRA.")

    # Model hparams
    parser.add_argument("--max_prompt_len", type=int, default=2048, help="Max length for prompt.") 
    parser.add_argument("--max_response_len", type=int, default=256, help="Max length for response.") 

    # System & Logging
    parser.add_argument("--device", type=str, default="cuda:2", help="Device to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log progress every X steps.") 
    parser.add_argument("--sample_output_steps", type=int, default=200, help="Output a sample every X steps.")
    parser.add_argument("--eval_every_epochs", type=int, default=1, help="Run evaluation every X epochs.")
    parser.add_argument("--save_every_epochs", type=int, default=1, help="Save a checkpoint every X epochs.")
    parser.add_argument("--save_intermediate_checkpoints", action='store_true',
                        help="Save intermediate checkpoints with epoch number.")
    
    # Generation hparams
    parser.add_argument("--max_new_tokens_eval", type=int, default=256, help="Max new tokens for evaluation.")
    parser.add_argument("--num_beams_eval", type=int, default=1, help="Beams for evaluation.")
    parser.add_argument("--do_sample_eval", action="store_true", help="Use sampling for evaluation.")
    parser.add_argument("--max_new_tokens_log", type=int, default=512, help="Max new tokens for logging.")
    parser.add_argument("--num_beams_log", type=int, default=1, help="Beams for logging.")
    parser.add_argument("--do_sample_log", action="store_true", help="Use sampling for logging.")

    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()
    
    if args.train_special_embeddings_in_sft and args.original_base_llm_vocab_size is None:
        parser.error("--original_base_llm_vocab_size is required when --train_special_embeddings_in_sft is set.")

    logger = logging.getLogger("Stage4_SFT_LoRA_Main") 
    if not logger.hasHandlers(): 
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])
    logger.info(f"Effective SFT batch size will be: {args.batch_size * args.gradient_accumulation_steps}")

    sft_lora(args)
