import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, scf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------参数区----------------------
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

TOKEN_PAIRS = [
    (209, 34),
    (7, 318),
    (146, 39)
    # 可以无限多对
]

FUNC_GROUP = "ether_O"     # 当前只对比这个官能团
SAMPLES_PER_TOKEN = 100      # 每个 Token 采样数量
MAX_ATOMS = 50
NUM_WORKERS = 8
BASIS, XC = "sto-3g", "hf"   # HF/STO-3G
DATA_PATH = "/home-ssd/Users/nsgm_zmx/Molecule/src_classification/Llama/property_analyze/data/CID2SMILES_special_tokens.csv"  # 你的数据路径
# ----------------------参数区----------------------

def sample_df(df, group, token, n=SAMPLES_PER_TOKEN):
    sub = df[(df["FUNC_GROUP"] == group) & (df["TOKEN_ID"] == token)]
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
    except Exception:
        return None
    return m3d

def calc_pi_electron_on_atom(mol, atom_idx, basis=BASIS, xc=XC):
    atom_list = []
    conf = mol.GetConformer()
    for a in mol.GetAtoms():
        pos = conf.GetAtomPosition(a.GetIdx())
        atom_list.append((a.GetSymbol(), (pos.x, pos.y, pos.z)))
    mol_p = gto.Mole()
    try:
        mol_p.build(atom=atom_list, basis=basis, verbose=0)
    except RuntimeError as e:
        print(f"[跳过] 构建分子失败: {e}")
        return None
    if xc.lower() == "hf":
        mf = scf.RHF(mol_p).run()
    else:
        mf = scf.RKS(mol_p).set(xc=xc).run()
    D = mf.make_rdm1()
    S = mol_p.intor("int1e_ovlp")
    P = D @ S
    ao_labels = mol_p.ao_labels(fmt=None)
    pz_idx = []
    target_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
    for i, (at, sym, l, m) in enumerate(ao_labels):
        if at == atom_idx and sym == target_symbol and l == "2p" and m == "z":
            pz_idx.append(i)
    if not pz_idx:
        return None
    occ = sum(D[i, i] * 2 for i in pz_idx)
    return occ

def process_one_atom(smiles, atom_idx, token_id):
    m3d = mol3d_with_conformer(smiles)
    if m3d is None:
        return None
    occ = calc_pi_electron_on_atom(m3d, atom_idx)
    if occ is None:
        return None
    return {
        "SMILES": smiles,
        "ATOM_IDX": atom_idx,
        "Token": f"Token{token_id}",
        "PI_ELECTRON_OCCUPANCY": occ
    }

def process_token(df_atoms, token_id):
    recs = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as exe:
        futures = [
            exe.submit(process_one_atom, row.SMILES, int(row.ATOM_IDX), token_id)
            for _, row in df_atoms.iterrows()
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Token{token_id}"):
            out = fut.result()
            if out:
                recs.append(out)
    return recs

def main():
    df = pd.read_csv(DATA_PATH)
    compiled = [(lab, Chem.MolFromSmarts(sma), idx) for lab, sma, idx in FUNC_SMARTS_LIST]
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="打标签"):
        mol = Chem.MolFromSmiles(row.SMILES)
        if mol is None: continue
        toks = list(map(int, str(row.TOKENS).split(',')))
        labels = ["other"] * mol.GetNumAtoms()
        used = [False] * mol.GetNumAtoms()
        for lab, patt, pick in compiled:
            for match in mol.GetSubstructMatches(patt):
                i = match[pick]
                if not used[i]:
                    labels[i] = lab
                    used[i] = True
        for i, lab in enumerate(labels):
            if lab != "other":
                records.append({
                    "SMILES": row.SMILES,
                    "ATOM_IDX": i,
                    "TOKEN_ID": toks[i],
                    "FUNC_GROUP": lab,
                    "N_ATOMS": mol.GetNumAtoms()
                })
    all_atoms = pd.DataFrame(records)
    all_atoms = all_atoms[all_atoms.N_ATOMS <= MAX_ATOMS]
    df_fg = all_atoms[all_atoms.FUNC_GROUP == FUNC_GROUP]
    print(f"{FUNC_GROUP} 匹配到的总原子数: {len(df_fg)}")

    for tokA, tokB in TOKEN_PAIRS:
        dfA = sample_df(df_fg, FUNC_GROUP, tokA, n=SAMPLES_PER_TOKEN)
        dfB = sample_df(df_fg, FUNC_GROUP, tokB, n=SAMPLES_PER_TOKEN)
        print(f"\n→ 处理 Pair ({tokA}, {tokB})，TokenA样本: {len(dfA)}，TokenB样本: {len(dfB)}")
        resultsA = process_token(dfA, tokA)
        resultsB = process_token(dfB, tokB)
        outA = pd.DataFrame(resultsA)
        outB = pd.DataFrame(resultsB)
        outA["PAIR"] = f"{tokA}_vs_{tokB}"
        outB["PAIR"] = f"{tokA}_vs_{tokB}"
        out = pd.concat([outA, outB], ignore_index=True)
        out.to_csv(f"{FUNC_GROUP}_token{tokA}_vs_token{tokB}_pi_electron.csv", index=False)
        print(f"已保存：{FUNC_GROUP}_token{tokA}_vs_token{tokB}_pi_electron.csv")

if __name__ == "__main__":
    main()
