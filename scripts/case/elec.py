import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, scf
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------- Parameter Section ----------------------
FUNC_SMARTS_LIST = [
    ("carboxy_C=O", "[CX3](=O)[OX1H0-,OX2H1]", 1),
    ("carboxy_OH",  "[CX3](=O)[OX1H0-,OX2H1]", 2),
    ("ketone_C=O",  "[#6][CX3](=O)[#6]",         2),
    ("aldehyde_C=O","[CX3H1](=O)[#6]",           1),
    ("ester_O",     "[CX3](=O)[OX2][#6]",        2),
    ("ether_O",     "[OD2]([#6])[#6]",           0),
    ("alcohol_OH",  "[OX2H][CX4]",               0),
    ("amine_NH2",   "[NX3;H2][CX4]",             0),
    ("halogen",     "[F,Cl,Br,I]",               0),
    ("alkyne_C",    "[CX2]#[CX1]",               0),
    ("alkene_C",    "[CX3]=[CX3]",               0),
    ("aromatic_C",  "c",                         0),
]
MIX_TOKEN = 24
SAMPLES_PER_GROUP = 100
MAX_ATOMS = 50
NUM_WORKERS = 10
BASIS, XC = "sto-3g", "hf"

TARGET_GROUPS = ["aromatic_C", "alcohol_OH"]
# ---------------------- Parameter Section ----------------------

def sample_df(df, group, token=None, n=SAMPLES_PER_GROUP):
    sub = df[df["FUNC_GROUP"] == group]
    if token is not None:
        sub = sub[sub["TOKEN_ID"] == token]
    return sub.head(n)

def mol3d_with_conformer(smiles):
    m2d = Chem.MolFromSmiles(smiles)
    if m2d is None: return None
    m3d = Chem.AddHs(m2d)
    if AllChem.EmbedMolecule(m3d, AllChem.ETKDG()) != 0: return None
    try:
        if AllChem.MMFFHasAllMoleculeParams(m3d):
            AllChem.MMFFOptimizeMolecule(m3d)
        else:
            AllChem.UFFOptimizeMolecule(m3d)
    except Exception: return None
    return m3d

def get_pz_ao_indices(mol_pyscf, atom_idx_rdkit):
    ao_labels = mol_pyscf.ao_labels(fmt=None)
    print("AO labels:", ao_labels)
    indices = []
    for i, ao in enumerate(ao_labels):
        print("AO:", ao)
        # Check ao structure
        # It might be that ao[0] is a string, like "0", you need int(ao[0]) == atom_idx_rdkit
        try:
            at_idx = int(ao[0]) if isinstance(ao[0], str) and ao[0].isdigit() else ao[0]
            sym = ao[1]
            l = ao[2]
            m = ao[3]
            print(f"at_idx={at_idx}, sym={sym}, l={l}, m={m}")
            if at_idx == atom_idx_rdkit and l == '2p' and m == 'z':
                if sym in ['C', 'O']:  # Focus on common main-group elements
                    indices.append(i)
        except Exception as e:
            print("Failed to parse AO label:", ao, e)
    #print("pz AO indices for atom", atom_idx_rdkit, indices)
    return indices

def calc_pi_electron_on_atom(mol, atom_idx, basis=BASIS, xc=XC):
    # Return the total electron occupancy on pz atomic orbitals of the target atom
    try:
        atom_list = []
        for atom in mol.GetAtoms():
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            atom_list.append((atom.GetSymbol(), (pos.x, pos.y, pos.z)))
        mol_pyscf = gto.Mole()
        mol_pyscf.build(atom=atom_list, basis=basis, verbose=0)
        mf = scf.RHF(mol_pyscf).run()
        dm = mf.make_rdm1()
        ao_idx = get_pz_ao_indices(mol_pyscf, atom_idx)
        if not ao_idx: return None
        # Compute the total occupancy across these pz orbitals
        occ = 0
        for idx in ao_idx:
            occ += dm[idx, idx] * 2   # Double occupancy
        return occ
    except Exception as e:
        # print("pi electron error", e)
        return None

def process_one_atom(smiles, atom_idx, typ, fg, token_id):
    m3d = mol3d_with_conformer(smiles)
    if m3d is None: return []
    pi_occ = calc_pi_electron_on_atom(m3d, atom_idx)
    if pi_occ is not None:
        return [{
            "SMILES": smiles,
            "ATOM_IDX": atom_idx,
            "TYPE": typ,
            "FUNC_GROUP": fg,
            "TOKEN_ID": token_id,
            "PI_ELECTRON_OCCUPANCY": pi_occ
        }]
    else:
        return []

def process_batch(df, typ, fg, token_id):
    args_list = [(row["SMILES"], int(row["ATOM_IDX"]), typ, fg, token_id) for _, row in df.iterrows()]
    records = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_one_atom, *args) for args in args_list]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"PiElectrons-{typ}-{fg}"):
            res = fut.result()
            if res: records.extend(res)
    return records

# 1. Label the dataset
df = pd.read_csv("/home-ssd/Users/nsgm_zmx/Molecule/src_classification/Llama/visualize/codeward/CID2SMILES_special_tokens.csv")
compiled = [(lab, Chem.MolFromSmarts(sma), idx) for lab, sma, idx in FUNC_SMARTS_LIST]
records = []
print("Labeling all atoms. This may take a while...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    smiles = row["SMILES"]
    tokens = list(map(int, str(row["TOKENS"]).split(',')))
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: continue
    n_atoms = mol.GetNumAtoms()
    atom_labels = ["other"] * n_atoms
    labeled_mask = [False] * n_atoms
    for lab, patt, pick in compiled:
        for mt in mol.GetSubstructMatches(patt):
            at_idx = mt[pick] if pick < len(mt) else None
            if at_idx is None: continue
            if not labeled_mask[at_idx]:
                atom_labels[at_idx] = lab
                labeled_mask[at_idx] = True
    for i in range(n_atoms):
        if atom_labels[i] == "other":
            continue
        records.append({
            "SMILES": smiles,
            "ATOM_IDX": i,
            "TOKEN_ID": tokens[i],
            "FUNC_GROUP": atom_labels[i],
            "N_ATOMS": n_atoms
        })
allatom_df = pd.DataFrame(records)
allatom_df = allatom_df[allatom_df["N_ATOMS"] <= MAX_ATOMS]
print(f"Total labeled atoms collected: {len(allatom_df)}")

mix_df = allatom_df[allatom_df["TOKEN_ID"] == MIX_TOKEN]
groups = mix_df["FUNC_GROUP"].value_counts().index.tolist()
if len(groups) < 2:
    raise ValueError("Mixed token must contain at least two functional groups.")
print(f"\nFunctional groups and counts within mixed token {MIX_TOKEN}:")
print(mix_df["FUNC_GROUP"].value_counts())

all_pi_records = []
for fg in TARGET_GROUPS:
    # MIXED
    mix_atoms = sample_df(mix_df, fg, MIX_TOKEN)
    print(f"Analyzing MIXED group {fg}, sample size {len(mix_atoms)}")
    all_pi_records += process_batch(mix_atoms, "MIXED", fg, MIX_TOKEN)
    # PURE
    freq = allatom_df[allatom_df["FUNC_GROUP"] == fg]["TOKEN_ID"].value_counts()
    pure_tok = freq.index[0]
    if pure_tok == MIX_TOKEN and len(freq) > 1:
        pure_tok = freq.index[1]
    pure_atoms = sample_df(allatom_df, fg, pure_tok)
    print(f"Analyzing PURE group {fg}, sample size {len(pure_atoms)}")
    all_pi_records += process_batch(pure_atoms, "PURE", fg, pure_tok)

out_df = pd.DataFrame(all_pi_records)
save_path = f"mix_token{MIX_TOKEN}_aromatic_C_hydroxy_O_pielectron_compare_{SAMPLES_PER_GROUP}.csv"
out_df.to_csv(save_path, index=False)
print(f"\nSaved π-electron occupancy results to: {save_path}")
print("All done!") 