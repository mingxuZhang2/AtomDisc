import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, dft
from tqdm import tqdm
from scipy.stats import ttest_ind, mannwhitneyu, ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed

# Functional group definitions
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

#MIX_TOKEN = int(input("Enter the mixed token ID to analyse: ").strip())
MIX_TOKEN = 39
SAMPLES_PER_GROUP = 100
BASIS, XC = "sto-3g", "hf"
MAX_ATOMS = 50
NUM_WORKERS = 10

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
fg1, fg2 = groups[:2]

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
    smiles, idx, typ, fg, token_id = args
    atom_list = mol3d_atom_list(smiles)
    if atom_list is None: return None
    try:
        mol = gto.Mole()
        mol.build(atom=atom_list, basis=BASIS, verbose=0)
        mf = dft.RKS(mol); mf.xc=XC; mf.kernel()
        charges = mf.mulliken_pop()[1]
        return {
            "TYPE": typ,
            "FUNC_GROUP": fg,
            "TOKEN_ID": token_id,
            "MULLIKEN_CHARGE": float(charges[idx])
        }
    except Exception:
        return None

def dft_multi_thread(rows, typ, fg, token_id):
    args_list = [(row["SMILES"], int(row["ATOM_IDX"]), typ, fg, token_id) for _, row in rows.iterrows()]
    results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(get_mulliken_charge, args) for args in args_list]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"DFT-{typ}-{fg}"):
            res = fut.result()
            if res is not None:
                results.append(res)
    return results

def sample_df(df, group, token=None, n=SAMPLES_PER_GROUP):
    # token=None indicates the entire group
    sub = df[df["FUNC_GROUP"]==group]
    if token is not None:
        sub = sub[sub["TOKEN_ID"]==token]
    return sub.head(n)

# === Sampling and DFT analysis ===
print("\nStart sampling and DFT analysis...")

# 1. Two functional groups inside the mixed token
dft_records = []
mix_fg1 = sample_df(mix_df, fg1, MIX_TOKEN)
mix_fg2 = sample_df(mix_df, fg2, MIX_TOKEN)
dft_records += dft_multi_thread(mix_fg1, "MIXED", fg1, MIX_TOKEN)
dft_records += dft_multi_thread(mix_fg2, "MIXED", fg2, MIX_TOKEN)

# 2. Both functional groups across the entire dataset
all_fg1 = sample_df(allatom_df, fg1)
all_fg2 = sample_df(allatom_df, fg2)
dft_records += dft_multi_thread(all_fg1, "ALL", fg1, -1)
dft_records += dft_multi_thread(all_fg2, "ALL", fg2, -1)

# 3. The most common "pure" token for each functional group
for fg in [fg1, fg2]:
    freq = allatom_df[allatom_df["FUNC_GROUP"] == fg]["TOKEN_ID"].value_counts()
    pure_tok = freq.index[0]
    if pure_tok == MIX_TOKEN and len(freq) > 1:
        pure_tok = freq.index[1]
    pure_df = sample_df(allatom_df, fg, pure_tok)
    dft_records += dft_multi_thread(pure_df, "PURE", fg, pure_tok)

dft_df = pd.DataFrame(dft_records)
save_path = f"mix_token{MIX_TOKEN}_{fg1}_{fg2}_dft_compare_500.csv"
dft_df.to_csv(save_path, index=False)
print(f"\nSaved DFT results: {save_path}")

# --- Statistical tests and visualization ---
def print_and_plot_dist(df, fg, mix_token, pure_token):
    plt.figure(figsize=(10,6))
    ax = sns.violinplot(
        x="TYPE", y="MULLIKEN_CHARGE", data=df[df["FUNC_GROUP"]==fg],
        order=["MIXED", "PURE", "ALL"], inner="quartile"
    )
    plt.title(f"DFT Distribution: {fg} in MixedToken {mix_token}, PureToken {pure_token}, All")
    plt.tight_layout()
    plt.savefig(f"mix_token{mix_token}_{fg}_dft_dist.png", dpi=120)
    plt.close()

for fg in [fg1, fg2]:
    print(f"\n==== Functional group {fg} ====")
    arr_mixed = dft_df[(dft_df["FUNC_GROUP"]==fg)&(dft_df["TYPE"]=="MIXED")]["MULLIKEN_CHARGE"]
    arr_all = dft_df[(dft_df["FUNC_GROUP"]==fg)&(dft_df["TYPE"]=="ALL")]["MULLIKEN_CHARGE"]
    arr_pure = dft_df[(dft_df["FUNC_GROUP"]==fg)&(dft_df["TYPE"]=="PURE")]["MULLIKEN_CHARGE"]
    pure_token = dft_df[(dft_df["FUNC_GROUP"]==fg)&(dft_df["TYPE"]=="PURE")]["TOKEN_ID"].iloc[0]
    print(f"  Mixed token vs ALL: T-test p={ttest_ind(arr_mixed, arr_all, equal_var=False).pvalue:.2e}, KS={ks_2samp(arr_mixed, arr_all).pvalue:.2e}")
    print(f"  PURE vs ALL:      T-test p={ttest_ind(arr_pure, arr_all, equal_var=False).pvalue:.2e}, KS={ks_2samp(arr_pure, arr_all).pvalue:.2e}")
    print_and_plot_dist(dft_df, fg, MIX_TOKEN, pure_token)

# Direct comparison between the two functional groups within the mixed token
arr1 = dft_df[(dft_df["FUNC_GROUP"]==fg1)&(dft_df["TYPE"]=="MIXED")]["MULLIKEN_CHARGE"]
arr2 = dft_df[(dft_df["FUNC_GROUP"]==fg2)&(dft_df["TYPE"]=="MIXED")]["MULLIKEN_CHARGE"]
print(f"\nMixed token comparison between two functional groups: T-test p={ttest_ind(arr1, arr2, equal_var=False).pvalue:.2e}, KS={ks_2samp(arr1, arr2).pvalue:.2e}")

# Direct comparison between the two functional groups across the full dataset
arr1_all = dft_df[(dft_df["FUNC_GROUP"]==fg1)&(dft_df["TYPE"]=="ALL")]["MULLIKEN_CHARGE"]
arr2_all = dft_df[(dft_df["FUNC_GROUP"]==fg2)&(dft_df["TYPE"]=="ALL")]["MULLIKEN_CHARGE"]
print(f"\nFull-dataset comparison between two functional groups: T-test p={ttest_ind(arr1_all, arr2_all, equal_var=False).pvalue:.2e}, KS={ks_2samp(arr1_all, arr2_all).pvalue:.2e}")

print("\nAll done!")