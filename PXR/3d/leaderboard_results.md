# PXR 3D Structure Track — Leaderboard Results

All submissions by jeremy (REDACTED_AFFILIATION).

## Submitted Rounds

| Round | File | Submitted | LDDT-PLI | BiSyRMSD | LDDT-LP | Coverage | Rank |
|-------|------|-----------|----------|----------|---------|----------|------|
| v2 | `submission_v2_8eqz_best.zip` | 2026-05-23 | 0.3852 | 4.9471 | 0.8922 | 1.0 | 31 |
| v4 | `submission_v4_roshambo_guided.zip` | 2026-05-24 | 0.3990 | 5.0913 | 0.8858 | 1.0 | 31 |
| v3 ¹ | `submission_v3_exhaust8.zip` | 2026-05-24 18:44 UTC | 0.3934 | 4.9894 | 0.8877 | 1.0 | 31 |

¹ Identity uncertain — leaderboard may have ordering/display issues per user note.

## Method Summary

| Round | Matched (108) | Novel (76) |
|-------|--------------|------------|
| v2 | Best(2o9i, 8eqz) by CNN score | GNINA → 2o9i, exhaustiveness=4 |
| v3 | Best(2o9i, 8eqz) by CNN score | GNINA → 2o9i, exhaustiveness=8 |
| v4 | Best(2o9i, 8eqz) by CNN score | GNINA → best ECFP4-matched template (64 PXR structures), exhaustiveness=4 |

## Score Comparison Across Rounds

| Round | LDDT-PLI | BiSyRMSD | LDDT-LP | vs v2 (LDDT-PLI) |
|-------|----------|----------|---------|------------------|
| v2 (8eqz best, exhaust=4) | 0.3852 | 4.9471 | 0.8922 | baseline |
| v3 (exhaust=8) ¹ | 0.3934 | 4.9894 | 0.8877 | +0.008 |
| v4 (64-template shape-guided) | 0.3990 | 5.0913 | 0.8858 | +0.014 |

**Best LDDT-PLI:** v4 (0.3990) — template diversity helps interaction quality  
**Best BiSyRMSD:** v2 (4.9471) — single receptor is more consistent  
**Best LDDT-LP:** v2 (0.8922) — 8eqz receptor gives cleanest ligand geometry

## Comparison to Leaderboard Neighbours (2026-05-25)

| Rank | Team | LDDT-PLI | BiSyRMSD | LDDT-LP |
|------|------|----------|----------|---------|
| 29 | florian-wuennemann | 0.4340 | 4.6989 | 0.9162 |
| 30 | damoluje | 0.4241 | 4.6384 | 0.8777 |
| 31 | **jeremy (best: v4)** | **0.3990** | 5.0913 | 0.8858 |

## Key Observations

- All three rounds score 0.39–0.40 LDDT-PLI. The gap to rank 29/30 (~0.43) is ~0.03–0.04 units.
- Higher exhaustiveness (v3 vs v2) gives marginal LDDT-PLI improvement (+0.008) with slightly worse BiSyRMSD.
- Template diversity (v4 vs v2) gives the best LDDT-PLI gain (+0.014) but at the cost of BiSyRMSD (+0.14 Å) — plausible since different receptor conformations shift the absolute coordinates.
- The main bottleneck is receptor conformation selection, not docking sampling. Ranks 29/30 likely use a method (AlphaFold3, better template routing, or experimental structure lookup) that places the ligand in the right pocket conformation.
- LDDT-LP is consistently high (0.886–0.892): ligand geometry is reasonable across all methods.
