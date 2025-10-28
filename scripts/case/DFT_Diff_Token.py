import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, dft
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Functional group SMARTS definitions ---
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
SAMPLES_PER_GROUP = 100
BASIS, XC = "sto-3g", "hf"
MAX_ATOMS = 50
NUM_WORKERS = 8

# IDs of the two tokens you want to analyse
tokA = 315
tokB = 490
tokens_of_interest = [tokA, tokB]

def mol3d_atom_list(smiles):
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
    conf = m3d.GetConformer()
    atom_list = []
    for atom in m3d.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atom_list.append((atom.GetSymbol(), (pos.x, pos.y, pos.z)))
    return atom_list

def get_mulliken_charge(args):
    smiles, idx, func_group, token_id = args
    atom_list = mol3d_atom_list(smiles)
    if atom_list is None: return None
    try:
        mol = gto.Mole()
        mol.build(atom=atom_list, basis=BASIS, verbose=0)
        mf = dft.RKS(mol); mf.xc=XC; mf.kernel()
        charges = mf.mulliken_pop()[1]
        return {
            "SMILES": smiles,
            "ATOM_IDX": idx,
            "FUNC_GROUP": func_group,
            "TOKEN_ID": token_id,
            "MULLIKEN_CHARGE": float(charges[idx])
        }
    except Exception:
        return None

def main():
    df = pd.read_csv("/home-ssd/Users/nsgm_zmx/Molecule/src_classification/Llama/property_analyze/data/CID2SMILES_special_tokens.csv")  # Replace with your path
    compiled = [(lab, Chem.MolFromSmarts(sma), pick) for lab, sma, pick in FUNC_SMARTS_LIST]
    records = []

    # 1. Assign functional group labels per atom and keep only the target tokens
    for _, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row["SMILES"]
        tokens = list(map(int, str(row["TOKENS"]).split(',')))
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: continue
        n_atoms = mol.GetNumAtoms()
        if n_atoms > MAX_ATOMS: continue
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
            tok = tokens[i]
            if atom_labels[i] == "other": continue
            if tok not in tokens_of_interest: continue  # Only analyse the target tokens
            records.append({
                "SMILES": smiles,
                "ATOM_IDX": i,
                "TOKEN_ID": tok,
                "FUNC_GROUP": atom_labels[i],
                "N_ATOMS": n_atoms
            })
    allatom_df = pd.DataFrame(records)
    print(f"Collected {len(allatom_df)} atoms with functional group and token annotations.")

    # 2. Sample each (token, functional group) combination and run DFT in parallel
    dft_records = []
    for fg in allatom_df["FUNC_GROUP"].unique():
        for tok in tokens_of_interest:
            sub_df = allatom_df[(allatom_df["FUNC_GROUP"]==fg) & (allatom_df["TOKEN_ID"]==tok)]
            group = sub_df.head(SAMPLES_PER_GROUP)
            args_list = [(row["SMILES"], int(row["ATOM_IDX"]), fg, tok) for _, row in group.iterrows()]
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = [executor.submit(get_mulliken_charge, args) for args in args_list]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"DFT-{fg}-token{tok}"):
                    res = fut.result()
                    if res is not None:
                        dft_records.append(res)
    dft_df = pd.DataFrame(dft_records)
    dft_df.to_csv(f"token{tokA}_vs_token{tokB}_funcgroup_dft.csv", index=False)
    print("Saved DFT results.")

if __name__ == "__main__":
    main()