"""
Headless version of 01_eda_curation.ipynb — runs without display.
Produces train_curated.csv, test_curated.csv, and diagnostic plots in results/.
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem, DataStructs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from utils import (
    COL_NAME, COL_PECSO, COL_SE, COL_SMILES,
    DATA_DIR,
    ecfp4_fp, mol_to_inchikey, remove_salts, smiles_to_mol,
)

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

# -----------------------------------------------------------------------
# 1. Load
# -----------------------------------------------------------------------
print("=== Loading data ===")
train = pd.read_csv(DATA_DIR / "train_raw.csv")
test  = pd.read_csv(DATA_DIR / "test_raw.csv")
print(f"Train: {train.shape}  Test: {test.shape}")
print(f"Train cols: {train.columns.tolist()}")
print(f"Test cols: {test.columns.tolist()}")

# -----------------------------------------------------------------------
# 2. pEC50 distribution
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
train[COL_PECSO].hist(bins=50, ax=axes[0], color="steelblue", edgecolor="white")
axes[0].set_xlabel("pEC50"); axes[0].set_title("pEC50 distribution (train)")
train[COL_SE].hist(bins=50, ax=axes[1], color="salmon", edgecolor="white")
axes[1].set_xlabel("pEC50 std.error"); axes[1].set_title("Measurement uncertainty")
plt.tight_layout()
plt.savefig(RESULTS / "pEC50_distribution.png", dpi=150)
plt.close()
print(f"\npEC50: min={train[COL_PECSO].min():.2f}  max={train[COL_PECSO].max():.2f}")
print(f"std.error: min={train[COL_SE].min():.3f}  max={train[COL_SE].max():.3f}")

# -----------------------------------------------------------------------
# 3. SMILES standardisation
# -----------------------------------------------------------------------
print("\n=== Standardising SMILES ===")

def standardise_df(df):
    df = df.copy()
    mols, canonical, flags = [], [], []
    for smi in df[COL_SMILES]:
        mol = smiles_to_mol(str(smi))
        if mol is None:
            mols.append(None); canonical.append(None); flags.append("invalid_smiles")
        else:
            mol = remove_salts(mol)
            mols.append(mol)
            canonical.append(Chem.MolToSmiles(mol))
            flags.append("ok")
    df["mol"] = mols
    df["canonical_smiles"] = canonical
    df["curation_flag"] = flags
    return df

train = standardise_df(train)
test  = standardise_df(test)
print(f"Train invalid: {(train['curation_flag'] != 'ok').sum()}")
print(f"Test  invalid: {(test['curation_flag']  != 'ok').sum()}")

# -----------------------------------------------------------------------
# 4. InChIKey deduplication
# -----------------------------------------------------------------------
print("\n=== Deduplication ===")
train["inchikey"] = train["mol"].apply(lambda m: mol_to_inchikey(m) if m else None)
test["inchikey"]  = test["mol"].apply(lambda m: mol_to_inchikey(m) if m else None)

before = len(train)
# Only deduplicate rows that have a valid InChIKey (None rows kept as-is)
has_key = train["inchikey"].notna()
train_keyed   = (train[has_key].sort_values(COL_SE)
                 .drop_duplicates(subset="inchikey", keep="first"))
train_no_key  = train[~has_key]
train = pd.concat([train_keyed, train_no_key], ignore_index=True)
print(f"Removed {before - len(train)} exact-InChIKey duplicates (kept 1 per key, lowest SE)")

test_keys = set(test["inchikey"].dropna())
print(f"Train molecules also in test (exact match): {train['inchikey'].isin(test_keys).sum()}")

# -----------------------------------------------------------------------
# 5. Quality filters
# -----------------------------------------------------------------------
print("\n=== Quality filters ===")
SE_THRESHOLD   = 0.5
PECSO_MIN      = 1.0
PECSO_MAX      = 8.5

mask_se      = train[COL_SE] > SE_THRESHOLD
mask_extreme = (train[COL_PECSO] < PECSO_MIN) | (train[COL_PECSO] > PECSO_MAX)
mask_invalid = train["curation_flag"] != "ok"

print(f"High std.error (>{SE_THRESHOLD}): {mask_se.sum()}")
print(f"Extreme pEC50:                    {mask_extreme.sum()}")
print(f"Invalid SMILES:                   {mask_invalid.sum()}")

train_curated = train[~(mask_se | mask_extreme | mask_invalid)].copy().reset_index(drop=True)
print(f"After quality filters: {len(train_curated)} / {len(train)}")

# -----------------------------------------------------------------------
# 6. Fingerprints for chemical space + AD filter
# -----------------------------------------------------------------------
print("\n=== Computing ECFP4 for chemical space analysis ===")

def fps_to_array(mols, n_bits=2048):
    arr = np.zeros((len(mols), n_bits), dtype=np.uint8)
    for i, m in enumerate(mols):
        if m is not None:
            fp = ecfp4_fp(m, n_bits)
            DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr

train_mols  = train_curated["mol"].tolist()
test_mols_ok = [m for m in test["mol"] if m is not None]

fp_train = fps_to_array(train_mols)
fp_test  = fps_to_array(test_mols_ok)

print(f"Train FP: {fp_train.shape}  Test FP: {fp_test.shape}")

# -----------------------------------------------------------------------
# 7. PCA
# -----------------------------------------------------------------------
print("Running PCA ...")
all_fps = np.vstack([fp_train, fp_test])
labels  = np.array(["train"] * len(fp_train) + ["test"] * len(fp_test))

pca = PCA(n_components=2, random_state=42)
coords_pca = pca.fit_transform(all_fps)
print(f"PCA variance explained: {pca.explained_variance_ratio_[:2].sum():.1%}")

df_pca = pd.DataFrame({
    "PC1": coords_pca[:, 0], "PC2": coords_pca[:, 1],
    "split": labels,
    "pEC50": list(train_curated[COL_PECSO].values) + [np.nan] * len(fp_test),
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for split, colour in [("train", "steelblue"), ("test", "tomato")]:
    sub = df_pca[df_pca["split"] == split]
    axes[0].scatter(sub["PC1"], sub["PC2"], c=colour, alpha=0.4, s=8, label=split)
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
axes[0].set_title("PCA — Train vs Test"); axes[0].legend()
tr = df_pca[df_pca["split"] == "train"]
sc = axes[1].scatter(tr["PC1"], tr["PC2"], c=tr["pEC50"], cmap="viridis", alpha=0.5, s=8)
plt.colorbar(sc, ax=axes[1], label="pEC50")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
axes[1].set_title("PCA — pEC50 activity")
plt.tight_layout()
plt.savefig(RESULTS / "chemical_space_pca.png", dpi=150)
plt.close()
print("Saved chemical_space_pca.png")

# -----------------------------------------------------------------------
# 8. t-SNE
# -----------------------------------------------------------------------
MAX_TSNE = 3000
if len(all_fps) > MAX_TSNE:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(all_fps), MAX_TSNE, replace=False)
    all_fps_sub, labels_sub = all_fps[idx], labels[idx]
else:
    all_fps_sub, labels_sub = all_fps, labels

print(f"Running t-SNE on {len(all_fps_sub)} molecules ...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
coords_tsne = tsne.fit_transform(all_fps_sub)

fig, ax = plt.subplots(figsize=(8, 6))
for split, colour in [("train", "steelblue"), ("test", "tomato")]:
    mask = labels_sub == split
    ax.scatter(coords_tsne[mask, 0], coords_tsne[mask, 1], c=colour, alpha=0.4, s=8, label=split)
ax.set_title("t-SNE — Train vs Test"); ax.legend()
plt.tight_layout()
plt.savefig(RESULTS / "chemical_space_tsne.png", dpi=150)
plt.close()
print("Saved chemical_space_tsne.png")

# -----------------------------------------------------------------------
# 9. Applicability Domain filter
# -----------------------------------------------------------------------
print("\n=== Applicability Domain (Tanimoto similarity to test) ===")

train_fps_rdkit = [ecfp4_fp(m) for m in train_mols]
test_fps_rdkit  = [ecfp4_fp(m) for m in test_mols_ok]

max_sim = np.array([
    max(DataStructs.BulkTanimotoSimilarity(fp, test_fps_rdkit))
    for fp in train_fps_rdkit
])
train_curated = train_curated.copy()
train_curated["max_sim_to_test"] = max_sim

print(f"Max Tanimoto to test: mean={max_sim.mean():.3f}  "
      f"median={np.median(max_sim):.3f}  min={max_sim.min():.3f}  max={max_sim.max():.3f}")

# Distribution plot with candidate thresholds
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(max_sim, bins=60, color="steelblue", edgecolor="white")
for thresh, col in [(0.10, "orange"), (0.15, "red"), (0.20, "purple")]:
    n = (max_sim < thresh).sum()
    ax.axvline(thresh, color=col, linestyle="--", label=f"thresh={thresh} (remove {n})")
ax.set_xlabel("Max Tanimoto similarity to test set")
ax.set_title("AD filter — Train→Test similarity")
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS / "ad_similarity_histogram.png", dpi=150)
plt.close()
print("Saved ad_similarity_histogram.png")

for thresh in [0.05, 0.10, 0.15, 0.20, 0.25]:
    n_rm = (max_sim < thresh).sum()
    print(f"  threshold={thresh:.2f}  remove={n_rm}  ({100*n_rm/len(max_sim):.1f}%)")

AD_THRESHOLD = 0.15
ad_mask = train_curated["max_sim_to_test"] < AD_THRESHOLD
print(f"\nApplying AD threshold={AD_THRESHOLD}: removing {ad_mask.sum()} molecules")
train_final_pre_chembl = train_curated[~ad_mask].reset_index(drop=True)
print(f"After AD filter: {len(train_final_pre_chembl)}")

# -----------------------------------------------------------------------
# 10. Save
# -----------------------------------------------------------------------
print("\n=== Saving ===")
keep_train = [COL_NAME, "canonical_smiles", COL_PECSO, COL_SE,
              "inchikey", "max_sim_to_test", "source"]
# only include cols that exist
keep_train = [c for c in keep_train if c in train_final_pre_chembl.columns]

out_train = train_final_pre_chembl[keep_train].copy()
out_train.rename(columns={"canonical_smiles": "SMILES"}, inplace=True)
out_train.to_csv(DATA_DIR / "train_curated.csv", index=False)

keep_test = [COL_NAME, "canonical_smiles", "inchikey"]
keep_test = [c for c in keep_test if c in test.columns]
out_test = test[keep_test].copy()
out_test.rename(columns={"canonical_smiles": "SMILES"}, inplace=True)
out_test.to_csv(DATA_DIR / "test_curated.csv", index=False)

print(f"Saved train_curated.csv: {len(out_train)} rows")
print(f"Saved test_curated.csv:  {len(out_test)} rows")

# -----------------------------------------------------------------------
# 11. Summary
# -----------------------------------------------------------------------
print("\n=== Curation Summary ===")
print(f"  Raw train              : {len(train)}")
print(f"  After dedup            : after")
print(f"  Removed high SE        : {mask_se.sum()}")
print(f"  Removed extreme pEC50  : {mask_extreme.sum()}")
print(f"  Removed invalid SMILES : {mask_invalid.sum()}")
print(f"  Removed low AD sim     : {ad_mask.sum()}")
print(f"  Final curated train    : {len(out_train)}")
print(f"  pEC50 range            : [{out_train[COL_PECSO].min():.2f}, {out_train[COL_PECSO].max():.2f}]")
