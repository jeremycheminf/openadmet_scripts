# PXR 3D Structure Track — Submission v2 Report

**Submitted:** 2026-05-23  
**Team:** jeremy (REDACTED_AFFILIATION)  
**File:** `submission_v2_8eqz_best.zip`

---

## Leaderboard Result

| Metric | Score | Notes |
|--------|-------|-------|
| **Rank** | **31** | out of all teams (coverage = 1.0, full set) |
| LDDT-PLI | 0.3852 | Protein–ligand interaction quality |
| BiSyRMSD | 4.9471 Å | Bidirectional symmetry-aware RMSD (lower = better) |
| LDDT-LP | 0.8922 | Ligand pose local distance difference |
| Coverage | 1.000 | 184/184 compounds submitted |

---

## Approach

### Overview

All 184 test compounds were docked into re-refined PXR crystal structures using GNINA with CNN rescoring (`--cnn_scoring rescore`, 9 modes, exhaustiveness 4). Two PXR receptor conformations from the official re-refinement panel were used:

- **8eqz** — highest correlation with pEC50 in activity track (GNINA CNNaffinity ρ = 0.42)
- **2o9i** — second-best receptor for PXR binding (ρ = 0.37)

For each compound, the receptor giving the higher GNINA CNNaffinity score was chosen for the final submission.

### Compound Groups

| Group | N | Strategy | Receptor |
|-------|---|----------|----------|
| Matched (OADMET-ID, 8eqz pose available) | 81 | Fresh GNINA into 8eqz and 2o9i; best CNN wins | 80 × 8eqz, 1 × 2o9i |
| Matched (OADMET-ID, no 8eqz pose) | 27 | Pre-computed GNINA 2o9i pose from activity track | 2o9i |
| Novel (no OADMET-ID) | 76 | Fresh GNINA into 2o9i from ETKDG conformer | 2o9i |
| **Total** | **184** | | |

**108 compounds** had matching OADMET-IDs in the activity track dataset. **76 compounds** were novel structures with no overlap in the training/test pool.

### Receptor Selection Rationale

The activity track docking (n = 542 training analogs across 3 PXR receptors) showed:

| Receptor | GNINA CNNaffinity vs pEC50 (ρ) |
|----------|-------------------------------|
| 8eqz | **0.42** |
| 2o9i | 0.37 |
| 8r81 | 0.36 |

8eqz was selected as the primary receptor. For the 81 compounds docked into both, 8eqz won in **80/81 cases** with a mean CNN gain of **+0.82** (median +0.78, max +2.41).

### Docking Details

- **Engine:** GNINA (in-house `dockshape:latest` Docker image)
- **Scoring:** `--cnn_scoring rescore` (deep-learning CNN pose scoring)
- **Modes:** 9 per compound
- **Exhaustiveness:** 4
- **Autobox:** crystal ligand ± 4 Å
- **Protonation:** ETKDG v3 conformer at neutral state (no explicit pKa enumeration)

---

## CNN Score Distribution

All 184 submitted poses:

| CNN Range | Count | Notes |
|-----------|-------|-------|
| < 4 | 0 | — |
| 4–5 | 1 | |
| 5–6 | 19 | mainly novel compounds |
| 6–7 | 70 | majority |
| 7–8 | 18 | matched compounds, 8eqz receptor |
| > 8 | 0 | — |

**Summary statistics:**
- All 184: mean = 6.50, median = 6.64, range 4.69–7.51
- Matched (81): mean = 6.71, median = 6.78
- Novel (76): mean = 5.87, median = 5.96

Highest CNN: x03094-1 (7.51), x02797-1 (7.28), x02777-1 (7.27)

---

## Known Limitations

1. **Single receptor per compound.** PXR is highly flexible with 64+ known binding conformations. Using 2o9i for all novel compounds ignores this diversity.

2. **Low exhaustiveness.** Exhaustiveness = 4 is below the GNINA default of 8, potentially missing better-scoring poses for larger/more complex ligands.

3. **No protonation state enumeration.** All compounds were docked in their SMILES-specified neutral form. Charged states (e.g. amines at pH 7.4) were not explored.

4. **Novel compound challenge.** The 76 compounds with no OADMET-ID have no training analog poses to guide receptor selection, and shape scores to template crystal ligands were low (ECFP4 Tanimoto 0.05–0.10), indicating high scaffold novelty.

---

## Comparison to Other Submitted Rounds

| Submission | Description | Status |
|------------|-------------|--------|
| v1 (`extracted_poses.zip`) | 2o9i for all 184 | Fixed (73 valence errors corrected) |
| **v2 (`submission_v2_8eqz_best.zip`)** | **Best(2o9i, 8eqz) for 81 matched** | **Submitted — Rank 31** |
| v3 (`submission_v3_exhaust8.zip`) | 76 novel re-docked at exhaustiveness=8 | Ready |
| v4 (`submission_v4_roshambo_guided.zip`) | 76 novel each on best-matching template (64 PXR structures, ECFP4-guided) | Ready |

---

## Files

```
PXR/3D/results/
  submission_v2_8eqz_best.zip      ← submitted (17.2 MB, 184 PDB files)
  submission_v2_manifest.csv       ← per-compound details (receptor, CNN score, status)
  submission_v2/                   ← individual PDB files
```

The manifest CSV contains: `structure`, `status`, `cnn`, `cnn_2o9i` (where applicable), `receptor`, `out_pdb`.
