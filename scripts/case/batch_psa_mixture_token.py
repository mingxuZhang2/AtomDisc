import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# --- CONFIGURATION ---
# Define all functional groups to be recognized.
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

# Define mixture tokens and the specific functional groups to compare for each.
# Format: { token_id: ("functional_group_1", "functional_group_2") }
MIXTURE_TOKEN_CONFIG = {
    39: ("ester_O", "ether_O"),
    24: ("alcohol_OH", "aromatic_C"),
    418: ("ether_O", "aromatic_C"),
    # 24 aromatic_C alcohol_OH
    # 39 ester_O ether_O
    # 418 ether_O aromatic_C
    # Add more tokens and their specific FG pairs here
}

# --- GLOBAL SETTINGS ---
SAMPLES_PER_GROUP = 500
MAX_ATOMS = 50
NUM_WORKERS = 10
PSA_RADIUS = 4

# --- HELPER & CALCULATION FUNCTIONS ---

def local_psa(mol, atom_idx, radius):
    """Calculates the local TPSA around a specific atom."""
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    amap = {}
    submol = Chem.PathToSubmol(mol, env, atomMap=amap)
    submol.UpdatePropertyCache()
    Chem.GetSymmSSSR(submol)
    try:
        return rdMolDescriptors.CalcTPSA(submol)
    except Exception:
        return None

def get_psa_feature(args):
    """Worker function for parallel PSA calculation."""
    smiles, idx, typ, fg, token_id, radius = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        psa = local_psa(mol, idx, radius)
        if psa is None: return None
        return {
            "SMILES": smiles,
            "ATOM_IDX": idx,
            "TYPE": typ,
            "FUNC_GROUP": fg,
            "TOKEN_ID": token_id,
            f"LOCAL_PSA_r{radius}": psa
        }
    except Exception:
        return None

def psa_multi_thread(rows, typ, fg, token_id, radius):
    """Manages multithreaded execution of PSA calculations."""
    args_list = [(row["SMILES"], int(row["ATOM_IDX"]), typ, fg, token_id, radius) for _, row in rows.iterrows()]
    results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(get_psa_feature, args) for args in args_list]
        desc = f"PSA Calc: Type='{typ}', FG='{fg}'"
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            res = fut.result()
            if res is not None:
                results.append(res)
    return results

def sample_df(df, group, token=None, n=SAMPLES_PER_GROUP):
    """Samples a dataframe for a given functional group and optional token."""
    sub = df[df["FUNC_GROUP"] == group]
    if token is not None:
        sub = sub[sub["TOKEN_ID"] == token]
    return sub.head(n)

# --- CORE ANALYSIS LOGIC ---

def analyze_mixture_token(mix_token, fg1, fg2, allatom_df, base_output_dir="mixture_analysis_results"):
    """
    Performs the full PSA comparison analysis for a single mixture token
    between two specified functional groups (fg1 and fg2).
    """
    print("\n" + "="*60)
    print(f"Analyzing Mixture Token: {mix_token}")
    print(f"Comparing specified functional groups: '{fg1}' vs '{fg2}'")
    print("="*60)

    os.makedirs(base_output_dir, exist_ok=True)

    mix_df = allatom_df[allatom_df["TOKEN_ID"] == mix_token]
    if mix_df.empty:
        print(f"Token {mix_token} not found in the dataset. Skipping.")
        return

    psa_records = []

    # 1. Analyze the two specified FGs within the mixture token
    mix_fg1 = sample_df(mix_df, fg1, mix_token)
    mix_fg2 = sample_df(mix_df, fg2, mix_token)
    psa_records += psa_multi_thread(mix_fg1, "MIXED", fg1, mix_token, PSA_RADIUS)
    psa_records += psa_multi_thread(mix_fg2, "MIXED", fg2, mix_token, PSA_RADIUS)

    # 2. Analyze the same two FGs from the entire dataset for baseline
    all_fg1 = sample_df(allatom_df, fg1)
    all_fg2 = sample_df(allatom_df, fg2)
    psa_records += psa_multi_thread(all_fg1, "ALL_DATASET", fg1, -1, PSA_RADIUS)
    psa_records += psa_multi_thread(all_fg2, "ALL_DATASET", fg2, -1, PSA_RADIUS)

    # 3. For each specified FG, find its most common "pure" token and analyze it
    for fg in [fg1, fg2]:
        freq = allatom_df[allatom_df["FUNC_GROUP"] == fg]["TOKEN_ID"].value_counts()
        if freq.empty:
            print(f"Warning: No data found for functional group '{fg}' in the dataset.")
            continue
        
        pure_tok = freq.index[0]
        if pure_tok == mix_token and len(freq) > 1:
            pure_tok = freq.index[1] # Take the second most common if the first is the mix token
        
        print(f"For FG '{fg}', most common 'pure' token is: {pure_tok}")
        pure_df = sample_df(allatom_df, fg, pure_tok)
        psa_records += psa_multi_thread(pure_df, "PURE", fg, pure_tok, PSA_RADIUS)

    if not psa_records:
        print("Analysis resulted in no valid PSA data. No output file generated.")
        return

    # Save the results for the current mixture token
    psa_df = pd.DataFrame(psa_records)
    save_path = os.path.join(base_output_dir, f"mix_token_{mix_token}_{fg1}_vs_{fg2}_psa_compare.csv")
    psa_df.to_csv(save_path, index=False)
    print(f"\n✅ Analysis for token {mix_token} complete. Results saved to: {save_path}")


def main():
    """Main function to load data and orchestrate batch analysis."""
    # 1. Load and label the entire dataset once
    try:
        df = pd.read_csv("/home-ssd/Users/nsgm_zmx/Molecule/src_classification/Llama/visualize/codeward/CID2SMILES_special_tokens.csv")
    except FileNotFoundError:
        print("Error: Input CSV file not found. Please check the path.")
        return
        
    compiled = [(lab, Chem.MolFromSmarts(sma), idx) for lab, sma, idx in FUNC_SMARTS_LIST]
    records = []

    print("Pre-processing and labeling all atoms, please wait...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row["SMILES"]
        try:
            tokens = list(map(int, str(row["TOKENS"]).split(',')))
        except (ValueError, AttributeError):
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() > MAX_ATOMS: continue
        
        n_atoms = mol.GetNumAtoms()
        atom_labels = {}
        for lab, patt, pick in compiled:
            for mt in mol.GetSubstructMatches(patt):
                at_idx = mt[pick]
                if at_idx not in atom_labels:
                    atom_labels[at_idx] = lab
        
        for i, lab in atom_labels.items():
            if i < len(tokens):
                records.append({
                    "SMILES": smiles,
                    "ATOM_IDX": i,
                    "TOKEN_ID": tokens[i],
                    "FUNC_GROUP": lab,
                })
    
    allatom_df = pd.DataFrame(records)
    print(f"Finished labeling. Found {len(allatom_df)} relevant atoms.")

    # 2. Loop through each mixture token configuration and run the analysis
    for mix_token, (fg1, fg2) in MIXTURE_TOKEN_CONFIG.items():
        analyze_mixture_token(mix_token, fg1, fg2, allatom_df)

    print("\n🎉 All batch analyses complete!")

if __name__ == "__main__":
    main()
