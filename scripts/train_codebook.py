import os
import json
import re
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from rdkit import Chem, RDLogger
from tqdm import tqdm
from einops import rearrange, repeat

from atomdisc.models.gnn import GNN
from atomdisc.utils.gnn_vq_utils import safe_parse_mol, fix_smiles, mol_to_graph

try:
    from atomdisc.config.stage1 import Stage1Config
except ImportError:
    Stage1Config = None

try:
    from vector_quantize_pytorch import VectorQuantize
except ImportError as exc:
    raise ImportError("vector_quantize_pytorch is required for codebook training") from exc

try:
    from torch_geometric.nn import MessagePassing, global_add_pool, global_max_pool, global_mean_pool
    from torch_geometric.utils import degree
except ImportError as exc:
    raise ImportError("torch_geometric is required for codebook training") from exc


def _default_stage1_config() -> "Stage1Config":
    if Stage1Config is not None:
        return Stage1Config()
    class _Cfg:
        num_layer = 5
        emb_dim = 300
        JK = "last"
        dropout_ratio = 0.1
        gnn_type = "gin"
        codebook_size = 512
    return _Cfg()


DEFAULT_STAGE1_CFG = _default_stage1_config()

GNN_MODEL_PATH = ""
SMILES_DATA_PATH = ""
VQ_CODEBOOK_SIZE = getattr(DEFAULT_STAGE1_CFG, "codebook_size", 512)
VQ_VAE_LEARNING_RATE = 1e-4
VQ_VAE_TRAIN_EPOCHS = 10
VQ_VAE_BATCH_SIZE = 128
RDLogger.DisableLog('rdApp.*')
VQ_VALIDATION_SUBSET_SIZE = 1000
PRECOMPUTED_EMBEDDINGS_FILE = "precomputed_gnn_embeddings.pt"
MAX_SAMPLES_FOR_KMEANS_INIT = 500_000


# --- Decoder Definition ---
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def extract_smiles_from_data(data_path):
    all_smiles_accumulator = []
    smiles_pattern = re.compile(r"<smiles>(.*?)</smiles>", re.IGNORECASE | re.DOTALL)
    print(f"Extracting SMILES from: {data_path}")
    invalid_smiles_count = 0
    parsed_smiles_count = 0
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data_item = json.loads(line)
                    texts_to_search_in = []
                    if 'text' in data_item and isinstance(data_item['text'], str): texts_to_search_in.append(data_item['text'])
                    if 'inputs' in data_item and isinstance(data_item['inputs'], str): texts_to_search_in.append(data_item['inputs'])
                    if 'output' in data_item and isinstance(data_item['output'], str): texts_to_search_in.append(data_item['output'])
                    if not texts_to_search_in: 
                        if isinstance(data_item, str): texts_to_search_in.append(data_item)
                        elif isinstance(data_item, dict): 
                            for value in data_item.values():
                                if isinstance(value, str): texts_to_search_in.append(value)
                    
                    for text_content in texts_to_search_in:
                        found_smiles = smiles_pattern.findall(text_content)
                        for s in found_smiles:
                            stripped_s = s.strip()
                            if stripped_s: 
                                mol = smiles_utils.safe_parse_mol(stripped_s)
                                if mol:
                                    all_smiles_accumulator.append(stripped_s)
                                    parsed_smiles_count +=1
                                else:
                                    if hasattr(smiles_utils, 'fix_smiles'):
                                        fixed_s = smiles_utils.fix_smiles(stripped_s)
                                        if fixed_s:
                                            mol_fixed = smiles_utils.safe_parse_mol(fixed_s)
                                            if mol_fixed:
                                                all_smiles_accumulator.append(fixed_s)
                                                parsed_smiles_count +=1
                                            else:
                                                invalid_smiles_count += 1
                                        else:
                                             invalid_smiles_count += 1
                                    else:
                                        invalid_smiles_count += 1
                except json.JSONDecodeError: print(f"Warning: Skipping malformed JSON line {i+1} in {data_path} : {line.strip()[:100]}")
                except Exception as e: print(f"Warning: Error processing line {i+1} in {data_path}: {e}. Line: {line.strip()[:100]}")
        
        unique_smiles_set = set(all_smiles_accumulator)
        sorted_unique_smiles = sorted(list(unique_smiles_set)) 
        print(f"Extracted {len(sorted_unique_smiles)} unique, valid, non-empty, sorted SMILES strings.")
        print(f"Found and successfully parsed {parsed_smiles_count} SMILES instances (before unique set).")
        if invalid_smiles_count > 0:
            print(f"Skipped {invalid_smiles_count} invalid/unparsable SMILES strings during extraction.")
        return sorted_unique_smiles
    except FileNotFoundError:
        print(f"Error: Data file not found: {data_path}"); return []

class AtomEmbeddingsDataset(Dataset):
    def __init__(self, embeddings_list):
        self.embeddings_list = embeddings_list 
    def __len__(self):
        return len(self.embeddings_list)
    def __getitem__(self, idx):
        return self.embeddings_list[idx]

def collate_atom_embeddings(batch_embeddings):
    padded_embeddings = pad_sequence(batch_embeddings, batch_first=True, padding_value=0.0)
    lengths = [emb.shape[0] for emb in batch_embeddings]
    attention_mask = torch.zeros(padded_embeddings.shape[0], padded_embeddings.shape[1], dtype=torch.bool)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1
    return padded_embeddings, attention_mask


def precompute_gnn_embeddings(gnn_model, all_smiles_strings, device, batch_size=128):
    print(f"\n--- Pre-computing GNN Embeddings for {len(all_smiles_strings)} SMILES ---")
    gnn_model.eval() 
        
    if os.path.exists(PRECOMPUTED_EMBEDDINGS_FILE):
        try:
            print(f"Loading precomputed GNN embeddings from {PRECOMPUTED_EMBEDDINGS_FILE}...")
            loaded_data = torch.load(PRECOMPUTED_EMBEDDINGS_FILE)
            
            if 'smiles' in loaded_data and 'embeddings' in loaded_data:
                loaded_smiles_list = loaded_data['smiles']
                if not isinstance(loaded_smiles_list, list) or not all(isinstance(s, str) for s in loaded_smiles_list):
                    print("  Loaded SMILES data is not a list of strings. Recomputing...")
                elif loaded_smiles_list == all_smiles_strings: 
                     print("  Loaded precomputed embeddings successfully (SMILES lists match).")
                     if 'embeddings_count' in loaded_data and loaded_data['embeddings_count'] == len(all_smiles_strings): # Check count
                        return [emb.cpu() for emb in loaded_data['embeddings']] 
                     else:
                        print(f"  SMILES lists match but embedding count differs in metadata (Loaded: {loaded_data.get('embeddings_count', -1)}, Expected: {len(all_smiles_strings)}). Recomputing...")
                else:
                    print("  Precomputed embeddings SMILES list mismatch. Recomputing...")
                    if len(loaded_smiles_list) != len(all_smiles_strings):
                        print(f"    Length differs: Loaded={len(loaded_smiles_list)}, Current={len(all_smiles_strings)}")
                    else:
                        print("    Lengths are the same, checking content for first mismatch...")
                        mismatch_found = False
                        for i in range(min(len(all_smiles_strings), len(loaded_smiles_list))): 
                            if loaded_smiles_list[i] != all_smiles_strings[i]:
                                print(f"      First mismatch at index {i}:")
                                print(f"        Loaded : '{loaded_smiles_list[i]}' (len: {len(loaded_smiles_list[i])})")
                                print(f"        Current: '{all_smiles_strings[i]}' (len: {len(all_smiles_strings[i])})")
                                mismatch_found = True
                                break 
                        if not mismatch_found and len(loaded_smiles_list) == len(all_smiles_strings): 
                             print("    No content mismatch found despite lists not being equal (this is unexpected if lengths are same).")
                    if len(all_smiles_strings) > 5 and len(loaded_smiles_list) > 5 : 
                         print("    Sample loaded_smiles_list[:3]:", loaded_smiles_list[:3])
                         print("    Sample all_smiles_strings[:3]:", all_smiles_strings[:3])
            else: 
                print("  Precomputed embeddings file is missing 'smiles' or 'embeddings' key. Recomputing...")
        except Exception as e:
            print(f"  Error loading precomputed embeddings: {e}. Recomputing...")

    precomputed_embeddings = []
    processed_smiles_for_embeddings = [] 
    with torch.no_grad():
        for i in tqdm(range(0, len(all_smiles_strings), batch_size), desc="Pre-computing GNN Embeddings"):
            current_batch_smiles = all_smiles_strings[i:i+batch_size] 
            for smiles_str in current_batch_smiles: 
                mol = smiles_utils.safe_parse_mol(smiles_str) 
                if mol:
                    graph_data = smiles_utils.mol_to_graph(mol)
                    if graph_data:
                        atom_features, edge_index, edge_attr = graph_data
                        if atom_features.size(0) > 0:
                             node_reps = gnn_model(atom_features.to(device), edge_index.to(device), edge_attr.to(device))
                             if node_reps.ndim == 2 and node_reps.size(0) > 0:
                                 precomputed_embeddings.append(node_reps.cpu())
                                 processed_smiles_for_embeddings.append(smiles_str) 
    
    print(f"Successfully pre-computed GNN embeddings for {len(precomputed_embeddings)} molecules.")
    if precomputed_embeddings: 
        try:
            print(f"Saving precomputed GNN embeddings to {PRECOMPUTED_EMBEDDINGS_FILE}...")
            torch.save({
                'smiles': processed_smiles_for_embeddings, # Save the SMILES list that corresponds to the embeddings
                'embeddings': precomputed_embeddings,
                'embeddings_count': len(processed_smiles_for_embeddings) # Add count for verification
            }, PRECOMPUTED_EMBEDDINGS_FILE)
            print("Saved precomputed embeddings.")
        except Exception as e:
            print(f"Error saving precomputed embeddings: {e}")
    return precomputed_embeddings


def validate_vq_codebook_usage(vq_model, embeddings_dataloader_val, device, codebook_size, description="Validating VQ Usage"):
    vq_model.eval()
    used_indices = set()
    total_atoms_processed_for_val = 0
    print(f"\nStarting VQ codebook usage validation ({description})...")
    with torch.no_grad():
        for padded_embeddings_batch, attention_mask_batch in tqdm(embeddings_dataloader_val, desc=description):
            padded_embeddings_batch = padded_embeddings_batch.to(device)
            vq_outputs = vq_model(padded_embeddings_batch) 
            if isinstance(vq_outputs, tuple) and len(vq_outputs) >= 2:
                indices_batch = vq_outputs[1] 
                actual_indices = indices_batch[attention_mask_batch.to(device)] 
                used_indices.update(actual_indices.cpu().tolist())
                total_atoms_processed_for_val += attention_mask_batch.sum().item()
            else:
                print("Warning: Could not get indices from VQ during validation.")
    num_unique_codes_used = len(used_indices)
    print(f"VQ Codebook Usage ({description}): {num_unique_codes_used} / {codebook_size} codes used from {total_atoms_processed_for_val} atoms.")
    return num_unique_codes_used

def run_gnn_vq_vae_training():
    if torch.cuda.is_available():
        effective_device = torch.device("cuda:0") 
        print(f"GNN/VQ-VAE processing on: {effective_device} (Original GPU 3, Name: {torch.cuda.get_device_name(0)})")
    else:
        effective_device = torch.device("cpu"); print("GNN/VQ-VAE processing on: CPU")

    print("\n--- Loading GNN Model ---")
    gnn_model = GNN(num_layer=gnn_vq_config.num_layer, emb_dim=gnn_vq_config.emb_dim, JK=gnn_vq_config.JK, drop_ratio=gnn_vq_config.dropout_ratio, gnn_type=gnn_vq_config.gnn_type)
    try:
        checkpoint = torch.load(GNN_MODEL_PATH, map_location=effective_device)
        state_dict_key = 'state_dict' if 'state_dict' in checkpoint else 'model_state_dict' 
        if state_dict_key not in checkpoint : state_dict_key = next(iter(checkpoint.keys())) if isinstance(checkpoint, dict) and checkpoint else None
        gnn_state_dict = checkpoint[state_dict_key] if state_dict_key and isinstance(checkpoint.get(state_dict_key), dict) else (checkpoint if isinstance(checkpoint, dict) else None)
        if gnn_state_dict is None: raise ValueError(f"Invalid GNN checkpoint: {GNN_MODEL_PATH}")
        cleaned_gnn_state_dict = {k.replace('module.', ''): v for k, v in gnn_state_dict.items()}
        load_result = gnn_model.load_state_dict(cleaned_gnn_state_dict, strict=False); print("GNN Missing/Unexpected keys:", load_result.missing_keys, load_result.unexpected_keys)
    except Exception as e: print(f"Error loading GNN: {e}"); return
    gnn_model.to(effective_device).eval()
    for param in gnn_model.parameters(): param.requires_grad = False
    print("GNN loaded, frozen, and in eval mode.")

    all_smiles_strings = extract_smiles_from_data(SMILES_DATA_PATH) 
    if not all_smiles_strings: print("No valid SMILES found. Exiting."); return

    gnn_atom_embeddings_list = precompute_gnn_embeddings(gnn_model, all_smiles_strings, effective_device, batch_size=256)
    if not gnn_atom_embeddings_list: print("No GNN embeddings precomputed. Exiting."); return
    
    print("\n--- Initializing VQ and Decoder Models ---")
    vq_model = VectorQuantize(dim=gnn_vq_config.emb_dim, codebook_size=VQ_CODEBOOK_SIZE, learnable_codebook=True, commitment_weight=0.25, kmeans_init=True, kmeans_iters=20, ema_update=False).to(effective_device) 
    
    print("Performing K-Means initialization for VQ codebook...")
    if hasattr(vq_model, '_codebook') and hasattr(vq_model._codebook, 'init_embed_') and hasattr(vq_model._codebook, 'initted') and vq_model._codebook.initted.item() == False:
        all_atom_reps_tensor_for_kmeans = torch.cat(gnn_atom_embeddings_list, dim=0)
        
        if all_atom_reps_tensor_for_kmeans.shape[0] > MAX_SAMPLES_FOR_KMEANS_INIT:
            print(f"  Total atom representations ({all_atom_reps_tensor_for_kmeans.shape[0]}) exceeds MAX_SAMPLES_FOR_KMEANS_INIT ({MAX_SAMPLES_FOR_KMEANS_INIT}).")
            print(f"  Randomly subsampling {MAX_SAMPLES_FOR_KMEANS_INIT} atoms for K-Means initialization.")
            indices = torch.randperm(all_atom_reps_tensor_for_kmeans.shape[0])[:MAX_SAMPLES_FOR_KMEANS_INIT]
            all_atom_reps_tensor_for_kmeans = all_atom_reps_tensor_for_kmeans[indices]
        
        all_atom_reps_tensor_for_kmeans = all_atom_reps_tensor_for_kmeans.to(effective_device)
        print(f"  Using {all_atom_reps_tensor_for_kmeans.shape[0]} atom representations for K-Means.")

        num_internal_codebooks = vq_model._codebook.num_codebooks
        if num_internal_codebooks == 1:
             all_atom_reps_for_init = rearrange(all_atom_reps_tensor_for_kmeans, 'n d -> 1 n d')
        else: 
             print(f"  VQ has {num_internal_codebooks} internal codebooks. K-Means will be run for each on the (subsampled) atom representations.")
             all_atom_reps_for_init = repeat(all_atom_reps_tensor_for_kmeans, 'n d -> h n d', h=num_internal_codebooks)

        with torch.no_grad(): 
             vq_model._codebook.init_embed_(all_atom_reps_for_init) 
        print("K-Means initialization completed.")
    elif hasattr(vq_model, '_codebook') and hasattr(vq_model._codebook, 'initted') and vq_model._codebook.initted.item() == True:
        print("VQ codebook already initialized.")
    else:
        print("K-Means init not performed: VQ codebook attributes for init not found or 'initted' is not False.")

    decoder_model = Decoder(input_dim=gnn_vq_config.emb_dim, hidden_dim=gnn_vq_config.emb_dim * 2, output_dim=gnn_vq_config.emb_dim).to(effective_device)
    print(f"Decoder initialized. VQ commitment_weight: {vq_model.commitment_weight if hasattr(vq_model, 'commitment_weight') else 'N/A'}")

    optimizer = AdamW(list(vq_model.parameters()) + list(decoder_model.parameters()), lr=VQ_VAE_LEARNING_RATE)
    print(f"Optimizer: AdamW with lr={VQ_VAE_LEARNING_RATE}")
    
    atom_embeddings_dataset = AtomEmbeddingsDataset(gnn_atom_embeddings_list)
    effective_batch_size = min(VQ_VAE_BATCH_SIZE, len(atom_embeddings_dataset))
    if effective_batch_size == 0: print("Dataset for VQ-VAE is empty. Exiting."); return
    embeddings_dataloader_train = DataLoader(atom_embeddings_dataset, batch_size=effective_batch_size, shuffle=True, collate_fn=collate_atom_embeddings, num_workers=4 if effective_device.type == 'cuda' else 0, pin_memory=True if effective_device.type == 'cuda' else False)

    print("\n--- Training VQ-VAE (VQ Codebook and Decoder) ---")
    for epoch in range(VQ_VAE_TRAIN_EPOCHS):
        vq_model.train(); decoder_model.train()
        epoch_total_loss, epoch_recon_loss, epoch_vq_loss = 0, 0, 0
        
        progress_bar = tqdm(embeddings_dataloader_train, desc=f"Epoch {epoch+1}/{VQ_VAE_TRAIN_EPOCHS}")
        for batch_idx, (original_embeddings_batch, attention_mask_batch) in enumerate(progress_bar):
            original_embeddings_batch = original_embeddings_batch.to(effective_device, non_blocking=True)
            attention_mask_batch = attention_mask_batch.to(effective_device, non_blocking=True) 

            optimizer.zero_grad()
            quantized_embeddings, indices, vq_c_loss = vq_model(original_embeddings_batch)
            reconstructed_embeddings = decoder_model(quantized_embeddings)

            recon_loss_unreduced = F.mse_loss(reconstructed_embeddings, original_embeddings_batch, reduction='none')
            mask_expanded = attention_mask_batch.unsqueeze(-1).expand_as(recon_loss_unreduced) 
            masked_recon_loss = (recon_loss_unreduced * mask_expanded).sum() / mask_expanded.sum().clamp(min=1e-6) 
            total_loss = masked_recon_loss + vq_c_loss 
            
            total_loss.backward()
            optimizer.step()

            epoch_total_loss += total_loss.item()
            epoch_recon_loss += masked_recon_loss.item()
            epoch_vq_loss += vq_c_loss.item()

            progress_bar.set_postfix({
                "Total L": f"{total_loss.item():.4f}", "Recon L": f"{masked_recon_loss.item():.4f}",
                "VQ L": f"{vq_c_loss.item():.4f}", "Avg Total L": f"{epoch_total_loss / (batch_idx + 1):.4f}"
            })
        
        avg_epoch_total_loss = epoch_total_loss / max(1, len(embeddings_dataloader_train)) 
        avg_epoch_recon_loss = epoch_recon_loss / max(1, len(embeddings_dataloader_train))
        avg_epoch_vq_loss = epoch_vq_loss / max(1, len(embeddings_dataloader_train))
        print(f"--- Epoch {epoch+1} Summary ---")
        print(f"Avg Total Loss: {avg_epoch_total_loss:.4f}, Avg Recon Loss: {avg_epoch_recon_loss:.4f}, Avg VQ Loss: {avg_epoch_vq_loss:.4f}")
        
        val_embeddings_list_subset = gnn_atom_embeddings_list[:min(len(gnn_atom_embeddings_list), VQ_VALIDATION_SUBSET_SIZE if VQ_VALIDATION_SUBSET_SIZE > 0 else len(gnn_atom_embeddings_list))]
        if val_embeddings_list_subset: 
            val_embeddings_dataset = AtomEmbeddingsDataset(val_embeddings_list_subset)
            val_dataloader_subset = DataLoader(val_embeddings_dataset, batch_size=effective_batch_size, shuffle=False, collate_fn=collate_atom_embeddings, num_workers=4 if effective_device.type == 'cuda' else 0, pin_memory=True if effective_device.type == 'cuda' else False)
            validate_vq_codebook_usage(vq_model, val_dataloader_subset, effective_device, VQ_CODEBOOK_SIZE, description="Subset Validation")
        else:
            print("Skipping VQ codebook usage validation on subset as validation subset is empty.")

    print("\n--- VQ-VAE Training Finished ---")
    
    print("\n--- Final VQ Codebook Usage Validation on Entire Dataset ---")
    full_embeddings_dataloader = DataLoader(atom_embeddings_dataset, batch_size=effective_batch_size, shuffle=False, collate_fn=collate_atom_embeddings, num_workers=4 if effective_device.type == 'cuda' else 0, pin_memory=True if effective_device.type == 'cuda' else False)
    validate_vq_codebook_usage(vq_model, full_embeddings_dataloader, effective_device, VQ_CODEBOOK_SIZE, description="Full Dataset Validation")

    combined_save_path = "./gnn_vq_decoder_model_trained_1024.pth"
    save_dict = {
        'gnn_config': { 
            'num_layer': gnn_vq_config.num_layer, 'emb_dim': gnn_vq_config.emb_dim,
            'JK': gnn_vq_config.JK, 'dropout_ratio': gnn_vq_config.dropout_ratio,
            'gnn_type': gnn_vq_config.gnn_type
        },
        'gnn_state_dict': gnn_model.state_dict(), # Save GNN state_dict
        'vq_state_dict': vq_model.state_dict(),
        'decoder_state_dict': decoder_model.state_dict(),
        'vq_codebook_size': VQ_CODEBOOK_SIZE,
        # 'gnn_model_path_used': GNN_MODEL_PATH # Path is less useful than state_dict if GNN is part of the saved artifact
    }
    torch.save(save_dict, combined_save_path)
    print(f"Trained VQ, Decoder, and GNN state_dict saved to {combined_save_path}")

if __name__ == "__main__":
    run_gnn_vq_vae_training()
