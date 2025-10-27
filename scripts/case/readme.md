# Case Study Scripts

This directory collects standalone analysis scripts used in the AtomDisc case studies. They operate on the tokenized atom–level CSV (e.g. `CID2SMILES_special_tokens.csv`) and compute local physicochemical descriptors for different AtomDisc tokens.

All scripts assume the input CSV contains at least the columns `SMILES`, `TOKENS` (comma-separated integer token sequence), and atom metadata exported from Stage 1. Update the default paths in each script (`INPUT_CSV_PATH`) to point to your local dataset before running.

## Common Requirements

- Python environment with RDKit, pandas, tqdm. DFT-based scripts additionally require PySCF.
- Input file: `CID2SMILES_special_tokens.csv` (default path hard-coded inside each script).
- Output: CSV files saved alongside the script or in a sub-directory, containing per-atom descriptor values and statistics.

## Local PSA Analyses

### `batch_PSA.py`
- **Goal**: Compute local topological polar surface area (TPSA) for atoms assigned to specified token pairs within a functional group.
- **Key parameters**:
  - `CONFIG`: dictionary listing functional groups (`smarts`, `pick_idx`) and token pairs `(tokA, tokB)` for comparison.
  - `PSA_RADIUS`: radius (in bond hops) defining the local subgraph used for TPSA (`CalcTPSA`).
  - `MAX_PER_TOKEN`: maximum samples collected per token per functional group (default 500).
  - `input_csv`: update `INPUT_CSV_PATH` in `main()`.
- **Usage**:
  ```bash
  python batch_PSA.py
  ```
  Creates a folder per functional group under `analysis_results/` and exports `tokA*_tokB*_localpsa_r{radius}.csv` along with console statistics (mean ± std, t-test, Mann–Whitney U-test).

### `batch_psa_mixture_token.py`
- **Goal**: Analyze “mixture tokens” that map to multiple functional groups by comparing their local PSA distributions against pure tokens and the full dataset baseline.
- **Key parameters**:
  - `FUNC_SMARTS_LIST`: SMARTS patterns used to label functional groups.
  - `MIXTURE_TOKEN_CONFIG`: mapping from token IDs to tuples `(functional_group_A, functional_group_B)` describing the mixture.
  - `SAMPLES_PER_GROUP`, `PSA_RADIUS`, `NUM_WORKERS`: sampling size, radius, and thread count for parallel PSA computation.
- **Usage**:
  ```bash
  python batch_psa_mixture_token.py
  ```
  Produces CSV files in `mixture_analysis_results/`, named `mix_token_{token}_{fg1}_vs_{fg2}_psa_compare.csv`.

## Mulliken Charge Calculations

### `DFT_Diff_Token.py`
- **Goal**: Compare Mulliken charges between two AtomDisc tokens across functional groups using single-point DFT calculations.
- **Key parameters**:
  - `tokens_of_interest`: list of token IDs to evaluate (defaults to `tokA=315`, `tokB=490`).
  - `FUNC_SMARTS_LIST`: functional group labeling rules.
  - `SAMPLES_PER_GROUP`: number of atoms sampled per `(token, functional group)` pair.
  - `BASIS`, `XC`: quantum chemistry settings (default `sto-3g`, Hartree–Fock).
  - Requires PySCF (`pyscf` package).
- **Usage**:
  ```bash
  python DFT_Diff_Token.py
  ```
  Outputs `token{tokA}_vs_token{tokB}_funcgroup_dft.csv` with columns `MULLIKEN_CHARGE`, `FUNC_GROUP`, `TOKEN_ID`, etc.

### `DFT_Mixture_Token.py`
- **Goal**: Evaluate Mulliken charge distributions for mixture tokens versus their dominant pure tokens.
- **Structure**: Similar to `DFT_Diff_Token.py`; inspect the script to adjust `MIX_TOKEN`, `TARGET_GROUPS`, or sampling parameters.
- **Usage**:
  ```bash
  python DFT_Mixture_Token.py
  ```

## π-Electron Occupancy Analyses

### `elec.py`
- **Goal**: Estimate π-electron occupancy on target atoms (e.g., aromatic vs. alcohol atoms) for a chosen mixture token by summing occupations of p<sub>z</sub> atomic orbitals.
- **Key parameters**:
  - `MIX_TOKEN`: token ID under investigation (default 24).
  - `TARGET_GROUPS`: list of functional groups pulled from the mixture and compared against their pure-token counterparts.
  - `SAMPLES_PER_GROUP`: sampling size per group.
  - `BASIS`, `XC`: PySCF settings (Hartree–Fock with `sto-3g`).
- **Usage**:
  ```bash
  python elec.py
  ```
  Saves `mix_token{MIX_TOKEN}_..._pielectron_compare_{SAMPLES_PER_GROUP}.csv` with per-atom π electron estimates.

### `elec_diff_token.py`
- **Goal**: Pairwise comparison of π-electron occupancies between two token IDs across functional groups.
- **Adjustments**: Set `TOKENS_OF_INTEREST`, `TARGET_GROUPS`, and PySCF parameters as needed.
- **Usage**:
  ```bash
  python elec_diff_token.py
  ```

## Tips

- Ensure the input CSV contains 3D-convertible molecules (the scripts build 3D coordinates via RDKit before running PySCF).
- Quantum chemistry steps can be time-consuming; consider reducing `SAMPLES_PER_GROUP` or `NUM_WORKERS` if resources are limited.
- Outputs can be aggregated and visualized using pandas, seaborn, or Jupyter notebooks for further case-study analysis.
