"""Utility helpers for Stage-1 GNN/VQ components and datasets."""

from __future__ import annotations

import json
import logging
import os
import random
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from atomdisc.config.stage1 import Stage1Config, DEFAULT_STAGE1_CONFIG
from atomdisc.models.gnn import GNN

try:
    from vector_quantize_pytorch import VectorQuantize
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError("vector_quantize_pytorch is required for AtomDisc utilities") from exc


SEED = 42
RDLogger.DisableLog("rdApp.*")


def set_seed(seed_value: int = SEED) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class GNNVQConfig:
    num_layer: int
    emb_dim: int
    JK: str
    dropout_ratio: float
    gnn_type: str
    codebook_size: int
    vq_decay: float = 0.8
    vq_commitment_weight: float = 1.0

    @classmethod
    def from_stage1(cls, stage1_cfg: Stage1Config | None = None) -> "GNNVQConfig":
        base = stage1_cfg or DEFAULT_STAGE1_CONFIG
        return cls(
            num_layer=base.num_layer,
            emb_dim=base.emb_dim,
            JK=base.JK,
            dropout_ratio=base.dropout_ratio,
            gnn_type=base.gnn_type,
            codebook_size=base.codebook_size,
        )


DEFAULT_CONFIG = GNNVQConfig.from_stage1()


allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)),
    "possible_chirality_list": [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    "possible_bonds": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "possible_bond_dirs": [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ],
}


def safe_parse_mol(smiles: str) -> Optional[Chem.Mol]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logging.debug("RDKit.MolFromSmiles returned None for %s", smiles)
            return None
        Chem.Kekulize(mol, clearAromaticFlags=True)
        return mol
    except Exception as exc:  # pragma: no cover - RDKit edge cases
        logging.debug("Exception parsing SMILES %s: %s", smiles, exc)
        return None


def fix_smiles(smiles: str) -> Optional[str]:
    fixed_smiles = re.sub(r"(?<!\\[)n(?![a-zA-Z])", "[nH]", smiles)
    return fixed_smiles if safe_parse_mol(fixed_smiles) is not None else None


def mol_to_graph(mol: Chem.Mol) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if mol is None:
        return None

    atom_features: list[list[int]] = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num == 0:
            logging.warning("Atom with atomic number 0 found. Skipping molecule.")
            return None
        try:
            atom_features.append([
                allowable_features["possible_atomic_num_list"].index(atomic_num),
                allowable_features["possible_chirality_list"].index(atom.GetChiralTag()),
            ])
        except ValueError as exc:
            logging.warning("Atom feature missing for atom %s: %s", atom.GetIdx(), exc)
            return None

    x = torch.tensor(np.array(atom_features), dtype=torch.long)
    if x.size(0) == 0:
        logging.warning("Molecule with no atoms encountered. Skipping.")
        return None

    edges: list[tuple[int, int]] = []
    edge_features: list[list[int]] = []
    if mol.GetNumBonds() > 0:
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            try:
                bond_type = allowable_features["possible_bonds"].index(bond.GetBondType())
                bond_dir = allowable_features["possible_bond_dirs"].index(bond.GetBondDir())
            except ValueError as exc:
                logging.warning("Bond feature missing for %s-%s: %s", i, j, exc)
                return None
            edges.extend([(i, j), (j, i)])
            edge_features.extend([[bond_type, bond_dir]] * 2)
    elif mol.GetNumAtoms() == 1:
        edges.append((0, 0))
        edge_features.append([0, 0])
    else:
        logging.warning("Molecule with %s atoms has no bonds. Skipping.", mol.GetNumAtoms())
        return None

    edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_features), dtype=torch.long)
    return x, edge_index, edge_attr


def load_gnn_vq_models(
    gnn_vq_checkpoint_path: str,
    device: torch.device,
    config: Optional[GNNVQConfig] = None,
) -> Tuple[GNN, VectorQuantize]:
    cfg = config or DEFAULT_CONFIG

    gnn = GNN(
        num_layer=cfg.num_layer,
        emb_dim=cfg.emb_dim,
        JK=cfg.JK,
        drop_ratio=cfg.dropout_ratio,
        gnn_type=cfg.gnn_type,
    ).to(device)

    vq = VectorQuantize(
        codebook_size=cfg.codebook_size,
        dim=cfg.emb_dim,
        decay=cfg.vq_decay,
        commitment_weight=cfg.vq_commitment_weight,
    ).to(device)

    if gnn_vq_checkpoint_path:
        checkpoint = torch.load(gnn_vq_checkpoint_path, map_location=device)
        gnn_state = checkpoint.get("gnn_state_dict") or checkpoint.get("model_state_dict") or checkpoint.get("model")
        if gnn_state:
            gnn.load_state_dict({k.replace("module.", ""): v for k, v in gnn_state.items()}, strict=False)
        else:
            logging.warning("GNN state_dict not found in checkpoint; using initialised weights.")

        vq_state = checkpoint.get("vq_state_dict")
        vq_codebook = checkpoint.get("vq_codebook") or checkpoint.get("vq2D")
        if vq_state:
            vq.load_state_dict({k.replace("module.", ""): v for k, v in vq_state.items()}, strict=False)
        elif vq_codebook is not None and hasattr(vq, "_codebook"):
            if vq._codebook.embed.data.shape != vq_codebook.shape:
                raise ValueError("VQ codebook shape mismatch")
            vq._codebook.embed.data.copy_(vq_codebook)
        else:
            logging.warning("VQ weights not found in checkpoint; using initialised codebook.")
    else:
        logging.warning("No gnn_vq_checkpoint_path provided; returning randomly initialised models.")

    gnn.eval()
    vq.eval()
    for param in gnn.parameters():
        param.requires_grad = False
    for param in vq.parameters():
        param.requires_grad = False
    return gnn, vq


class MoleculeTextDataset(Dataset):
    def __init__(self, file_path: str):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        self.items: List[str] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text_content = data.get("text") or data.get("annotated_response")
                if not text_content:
                    continue
                processed = text_content.split("\n\n")[0].strip()
                if processed:
                    self.items.append(processed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> str:
        return self.items[idx]


class MolVQCollator:
    def __init__(self, tokenizer: AutoTokenizer, gnn: GNN, vq: VectorQuantize, device: torch.device, max_seq_length: int = 1024):
        self.tokenizer = tokenizer
        self.gnn = gnn
        self.vq = vq
        self.device = device
        self.max_seq_length = max_seq_length
        self.smiles_pattern = re.compile(r"<smiles>(.*?)</smiles>", flags=re.DOTALL)

        if self.tokenizer.pad_token is None:
            logging.warning("Tokenizer missing pad_token. Setting it to eos_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, batch_texts: List[str]) -> dict:
        batch_input_ids = []
        batch_attention_masks = []
        batch_labels = []
        batch_vq_losses = []
        batch_vq_indices_flat: list[int] = []

        for text_item in batch_texts:
            processed_text = text_item
            matches = list(self.smiles_pattern.finditer(text_item))
            for match in reversed(matches):
                smiles_string = match.group(1).strip()
                mol = safe_parse_mol(smiles_string) or (lambda fixed=fix_smiles(smiles_string): safe_parse_mol(fixed) if fixed else None)()

                replacement = ""
                current_vq_loss = None
                if mol:
                    graph = mol_to_graph(mol)
                    if graph:
                        atom_features, edge_index, edge_attr = [t.to(self.device) for t in graph]
                        if atom_features.size(0) >= 2 or not self.gnn.training:
                            representations = self.gnn(atom_features, edge_index, edge_attr)
                            _, quantized_codes, vq_loss_output = self.vq(representations)
                            batch_vq_indices_flat.extend(quantized_codes.cpu().tolist())
                            if isinstance(vq_loss_output, dict):
                                current_vq_loss = vq_loss_output.get("loss", torch.tensor(0.0, device=self.device))
                            elif isinstance(vq_loss_output, torch.Tensor):
                                current_vq_loss = vq_loss_output
                            tokens = " ".join(f"<atom_{code.item()}>" for code in quantized_codes)
                            replacement = f"<mol> {tokens} </mol>"
                if current_vq_loss is not None:
                    batch_vq_losses.append(current_vq_loss)
                processed_text = processed_text[: match.start()] + replacement + processed_text[match.end():]

            encoded = self.tokenizer(
                processed_text,
                max_length=self.max_seq_length,
                padding=False,
                truncation=True,
                return_tensors=None,
                return_attention_mask=True,
            )
            input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
            attention = torch.tensor(encoded["attention_mask"], dtype=torch.long)
            labels = input_ids.clone()

            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention)
            batch_labels.append(labels)

        final_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        final_attention_masks = pad_sequence(batch_attention_masks, batch_first=True, padding_value=0)
        final_labels = pad_sequence(batch_labels, batch_first=True, padding_value=-100)
        final_labels[final_labels == self.tokenizer.pad_token_id] = -100
        if self.tokenizer.bos_token_id is not None:
            final_labels[final_labels == self.tokenizer.bos_token_id] = -100

        mean_vq_loss = (
            torch.stack([loss.to(self.device) for loss in batch_vq_losses]).mean()
            if batch_vq_losses
            else torch.tensor(0.0, device=self.device)
        )

        return {
            "input_ids": final_input_ids,
            "attention_mask": final_attention_masks,
            "labels": final_labels,
            "vq_loss": mean_vq_loss,
            "vq_indices": batch_vq_indices_flat,
        }


def get_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
