import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from tqdm import tqdm
from scipy.stats import ttest_ind, mannwhitneyu
from multiprocessing import Pool, cpu_count
import os
from functools import partial

# --- CONFIGURATION ---
# Define all functional groups and token pairs you want to analyze here.
# fg_name: A descriptive name for the functional group. This will also be the directory name.
#   - smarts: The SMARTS string for the functional group.
#   - pick_idx: The index of the atom of interest within the SMARTS match.
#   - token_pairs: A list of tuples, where each tuple is a pair of token IDs to compare.
CONFIG = {
    "alcohol_OH": {
        "smarts": "[OX2H][CX4]",
        "pick_idx": 0,
        "token_pairs": [
            (268, 243),
            (374, 379),
            (511, 423),
        ],
    },
}

# --- GLOBAL SETTINGS ---
MAX_ATOMS = 50
MAX_PER_TOKEN = 500  # Max samples per token for a given pair
PSA_RADIUS = 4       # Radius for local PSA calculation

# --- HELPER FUNCTIONS (No changes needed here) ---

def mol3d(smiles:str):
    """Generates a 3D conformation for a SMILES string."""
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

def calc_local_psa(smiles, atom_idx, radius):
    """Calculates the local TPSA around a specific atom within a given radius."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    amap = {}
    submol = Chem.PathToSubmol(mol, env, atomMap=amap)
    submol.UpdatePropertyCache()
    Chem.GetSymmSSSR(submol)
    try:
        return rdMolDescriptors.CalcTPSA(submol)
    except Exception:
        return None

def process_one(row, radius):
    """Worker function for multiprocessing. Calculates local PSA for a single row."""
    smi, idx, tok = row["SMILES"], int(row["ATOM_IDX"]), row["TOKEN_ID"]
    val = calc_local_psa(smi, idx, radius)
    if val is not None:
        return {"SMILES": smi, "ATOM_IDX": idx, "TOKEN_ID": tok, f"LOCAL_PSA_r{radius}": val}
    return None

# --- CORE ANALYSIS LOGIC ---

def analyze_token_pair(fg_name, fg_config, tokA, tokB, all_mols_df, base_output_dir):
    """
    Analyzes a single pair of tokens for a given functional group.
    """
    print("-" * 50)
    print(f"Analyzing FG: '{fg_name}', Token Pair: ({tokA}, {tokB})")

    # Create a directory for the functional group if it doesn't exist
    fg_dir = os.path.join(base_output_dir, fg_name)
    os.makedirs(fg_dir, exist_ok=True)

    patt = Chem.MolFromSmarts(fg_config["smarts"])
    pick_idx = fg_config["pick_idx"]
    records = []

    # Iterate through all molecules to find matches for the current functional group
    for _, row in tqdm(all_mols_df.iterrows(), total=len(all_mols_df), desc=f"Scanning for {fg_name}"):
        smiles = row["SMILES"]
        try:
            tokens = list(map(int, str(row["TOKENS"]).split(',')))
        except (ValueError, AttributeError):
            continue # Skip rows with invalid token format
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() > MAX_ATOMS:
            continue
        
        # Find all occurrences of the functional group in the molecule
        for match in mol.GetSubstructMatches(patt):
            atom_of_interest_idx = match[pick_idx]
            if atom_of_interest_idx < len(tokens):
                token_at_idx = tokens[atom_of_interest_idx]
                if token_at_idx in (tokA, tokB):
                    records.append({"SMILES": smiles, "ATOM_IDX": atom_of_interest_idx, "TOKEN_ID": token_at_idx})

    if not records:
        print(f"No molecules found for FG '{fg_name}' with tokens {tokA} or {tokB}. Skipping.")
        return

    env_df = pd.DataFrame(records)
    dfA = env_df[env_df["TOKEN_ID"] == tokA].head(MAX_PER_TOKEN)
    dfB = env_df[env_df["TOKEN_ID"] == tokB].head(MAX_PER_TOKEN)
    
    if dfA.empty or dfB.empty:
        print(f"Could not find sufficient samples for both tokens {tokA} and {tokB}. Skipping pair.")
        return

    combined_df = pd.concat([dfA, dfB]).reset_index(drop=True)
    print(f"Found {len(combined_df)} samples: {len(dfA)} for tokA={tokA}, {len(dfB)} for tokB={tokB}")

    # Use multiprocessing to calculate local PSA
    # functools.partial is used to pass the fixed 'radius' argument to the worker function
    process_func = partial(process_one, radius=PSA_RADIUS)
    with Pool(min(cpu_count(), 16)) as pool:
        rows_to_process = [row for _, row in combined_df.iterrows()]
        out = list(tqdm(pool.imap_unordered(process_func, rows_to_process), total=len(rows_to_process), desc="Calculating PSA"))
    
    df_out = pd.DataFrame([x for x in out if x is not None])
    
    if df_out.empty:
        print("PSA calculation resulted in no valid data. Skipping pair.")
        return

    output_csv = os.path.join(fg_dir, f"tokA{tokA}_tokB{tokB}_localpsa_r{PSA_RADIUS}.csv")
    df_out.to_csv(output_csv, index=False)
    print(f"\n✅ Local PSA data saved to {output_csv}")

    # Perform and print statistical analysis
    valsA = df_out[df_out["TOKEN_ID"] == tokA][f"LOCAL_PSA_r{PSA_RADIUS}"]
    valsB = df_out[df_out["TOKEN_ID"] == tokB][f"LOCAL_PSA_r{PSA_RADIUS}"]
    
    print(f"\n--- Stats for {fg_name} ({tokA} vs {tokB}) ---")
    print(f"Token {tokA} ({len(valsA)} samples) Mean PSA: {valsA.mean():.4f} ± {valsA.std():.4f}")
    print(f"Token {tokB} ({len(valsB)} samples) Mean PSA: {valsB.mean():.4f} ± {valsB.std():.4f}")

    if len(valsA) > 1 and len(valsB) > 1:
        t, p = ttest_ind(valsA, valsB, equal_var=False)
        u, u_p = mannwhitneyu(valsA, valsB, alternative='two-sided')
        print(f"T-test: t={t:.4f}, p={p:.3e}")
        print(f"Mann-Whitney U: U={u:.2f}, p={u_p:.3e}")
    print("-" * 50 + "\n")


def main(input_csv, base_output_dir="analysis_results"):
    """
    Main function to orchestrate the analysis based on the CONFIG.
    """
    print("Loading molecule data...")
    try:
        all_mols_df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv}")
        return
    
    print(f"Loaded {len(all_mols_df)} molecules.")

    # Iterate through each functional group defined in the config
    for fg_name, fg_config in CONFIG.items():
        # Iterate through each token pair for the current functional group
        for tokA, tokB in fg_config["token_pairs"]:
            analyze_token_pair(
                fg_name=fg_name,
                fg_config=fg_config,
                tokA=tokA,
                tokB=tokB,
                all_mols_df=all_mols_df.copy(), # Pass a copy to be safe
                base_output_dir=base_output_dir
            )
    print("\n🎉 All analyses complete.")


if __name__ == "__main__":
    # Set the path to your input CSV file
    INPUT_CSV_PATH = "/home-ssd/Users/nsgm_zmx/Molecule/src_classification/Llama/visualize/codeward/CID2SMILES_special_tokens.csv"
    main(input_csv=INPUT_CSV_PATH)
