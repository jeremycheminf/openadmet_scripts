"""Download all input data: the challenge's own HF train/test set, plus the
PubChem AID 1851 auxiliary qHTS cytochrome panel used by the multitask fine-tune.

Run from the repo root:  python scripts/01_download_data.py
"""

from io import StringIO
import time

import pandas as pd
import requests
from cyp_submission.data import HF_DIR, load_test_blinded, load_train_inhibition
from cyp_submission.paths import DATA_EXTERNAL, ensure_dirs
from huggingface_hub import snapshot_download

AID = 1851
ACCESSION_TO_ISO = {
    "NP_000752": "CYP1A2", "NP_000760": "CYP2C19", "NP_000762": "CYP2C9",
    "NP_001020332": "CYP2D6", "NP_059488": "CYP3A4",
}
CONCISE_URL = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{AID}/concise/CSV"
FULL_URL = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{AID}/CSV"
PROPERTY_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/CanonicalSMILES/CSV"
SID_BATCH, CID_BATCH = 3000, 200


def download_hf_train_test() -> None:
    HF_DIR.parent.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(repo_id="openadmet/cyp-challenge-train-test", repo_type="dataset",
                              local_dir=str(HF_DIR))
    print(f"HF train/test ready at {path}")


def download_pubchem_qhts() -> None:
    """PubChem AID 1851 (NCATS qHTS cytochrome panel, ~16.5k diversity-library
    compounds). Full per-SID CSV (batched under its 10k-SID cap) carries continuous
    potency (Fit_LogAC50); the bulk 'concise' export only has binary outcomes."""
    out_path = DATA_EXTERNAL / "pubchem_cyp_qhts_aid1851.csv"
    if out_path.exists():
        print(f"{out_path} already exists, skipping")
        return

    concise = pd.read_csv(StringIO(requests.get(CONCISE_URL, timeout=180).text))
    sids = concise["SID"].unique().tolist()
    print(f"AID {AID}: {len(sids)} unique SIDs, {concise['CID'].nunique()} unique CIDs")

    frames = []
    for i in range(0, len(sids), SID_BATCH):
        batch = sids[i:i + SID_BATCH]
        resp = requests.post(FULL_URL, data={"sid": ",".join(map(str, batch))}, timeout=120)
        resp.raise_for_status()
        frames.append(pd.read_csv(StringIO(resp.text), skiprows=[1, 2]))
        print(f"  dose-response: {min(i + SID_BATCH, len(sids))}/{len(sids)}")
        time.sleep(0.3)
    full = pd.concat(frames, ignore_index=True)

    full = full.rename(columns={"PUBCHEM_CID": "CID", "Panel Target": "target_accession",
                                 "PUBCHEM_ACTIVITY_OUTCOME": "outcome"})
    full["iso"] = full["target_accession"].str.split(".").str[0].map(ACCESSION_TO_ISO)
    full = full.dropna(subset=["iso", "CID"])
    full["CID"] = full["CID"].astype(int)
    full["pIC50_like"] = -pd.to_numeric(full["Fit_LogAC50"], errors="coerce")

    potency = full.pivot_table(index="CID", columns="iso", values="pIC50_like", aggfunc="mean")
    binary = full.assign(active=full["outcome"].map({"Active": 1.0, "Inactive": 0.0})).pivot_table(
        index="CID", columns="iso", values="active", aggfunc="first")

    cids = potency.index.tolist()
    print(f"fetching SMILES for {len(cids)} CIDs...")
    smi_frames = []
    for i in range(0, len(cids), CID_BATCH):
        batch = cids[i:i + CID_BATCH]
        resp = requests.post(PROPERTY_URL, data={"cid": ",".join(map(str, batch))}, timeout=60)
        resp.raise_for_status()
        smi_frames.append(pd.read_csv(StringIO(resp.text)))
        time.sleep(0.25)
    smiles_df = pd.concat(smi_frames, ignore_index=True).drop_duplicates(subset="CID").set_index("CID")

    df = potency.add_suffix("_pIC50").join(binary.add_suffix("_active"), how="outer").join(
        smiles_df, how="inner")
    df = df.dropna(subset=["ConnectivitySMILES"]).reset_index().rename(
        columns={"ConnectivitySMILES": "smiles"})

    challenge_smiles = set(load_train_inhibition()["SMILES"]) | set(load_test_blinded()["SMILES"])
    n_before = len(df)
    df = df[~df["smiles"].isin(challenge_smiles)].reset_index(drop=True)
    print(f"dropped {n_before - len(df)} exact-SMILES matches with challenge train/test")
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}  ({len(df)} rows)")


def main() -> None:
    ensure_dirs()
    download_hf_train_test()
    download_pubchem_qhts()


if __name__ == "__main__":
    main()
