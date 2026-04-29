# OpenADMET PXR Structure Baseline Submission Report

## Objective

This report describes the baseline structure-submission workflow used to prepare a blind-test submission package for the OpenADMET PXR structure task. The report is restricted to the completed baseline submission and is written as a methods/results summary of the submitted baseline only.

## Dataset

- Input set: official blinded structure test set
- Input rows: `184`
- Input columns: `['structure', 'smiles']`
- Baseline package exists: `yes`
- Baseline package PDB count: `184`
- Baseline package size (bytes): `8349328`

## Receptor Template Panel

Human holo PXR structures were curated into a monomeric receptor panel with associated reference ligands and docking boxes. Exact template-ligand duplicates were retained for curation bookkeeping, but template routing used the deduplicated selected subset.

Template panel:
- `1ILH` (PDB `1ILH`, chain `A`, ligand `SRL`, resolution `2.76` A): curated-but-not-used-for-assignment
- `1M13` (PDB `1M13`, chain `A`, ligand `HYF`, resolution `2.15` A): selected
- `1NRL` (PDB `1NRL`, chain `A`, ligand `SRL`, resolution `2.0` A): selected
- `3HVL` (PDB `3HVL`, chain `A`, ligand `SRL`, resolution `2.1` A): curated-but-not-used-for-assignment
- `3R8D` (PDB `3R8D`, chain `A`, ligand `PNU`, resolution `2.8` A): selected

## Ligand Preparation And Template Assignment

Ligands were read from the official blinded structure CSV and normalised into an internal ligand identifier plus SMILES representation using the `structure` identifier column. For 2D inputs, protonation-state preparation was applied before docking at pH 7.4, and the dominant prepared state was used for 3D docking calculations.

Template assignment used 2D ligand similarity against template co-crystal ligands:

- fingerprint: Morgan / ECFP4-like radius 2, 2048 bits
- similarity metric: Tanimoto
- top template: always retained
- second template: retained only when within 0.10 Tanimoto of the best match and chemically non-redundant relative to the first template

Assignment summary:

- completed assignment rows: `368`
- average templates per ligand: `2.00`

## Docking And Pose Refinement

Each assigned ligand-template pair was processed with the following baseline docking workflow:

1. UniDock-Pro hybrid docking in `detail` mode
2. search exhaustiveness: `8`
3. maximum poses per ligand-template pair: `3`
4. GNINA pose minimisation with `--minimize`
5. GNINA CNN rescoring with `--cnn_scoring rescore`
6. GNINA autoboxing from the template reference ligand

Docking summary:

- completed GNINA pose rows: `973`
- average retained poses per assignment: `2.64`

## Baseline Pose Selection

The blind-test baseline package was generated directly from the GNINA-minimised and CNN-rescored poses. One final complex was selected per ligand using the following ranking rule:

1. highest GNINA `CNNaffinity`
2. tie-break by most favourable GNINA `minimizedAffinity`
3. tie-break by higher ligand-template similarity
4. tie-break by lower pose rank

Each selected ligand pose was merged with the exact monomeric receptor PDB used in docking to produce one protein-ligand PDB per ligand.

Selection summary:

- baseline complexes written: `184`
- passed basic local PDB validation: `184`
- mean CNNaffinity: `6.100`
- median CNNaffinity: `6.132`
- mean minimizedAffinity: `-7.156`
- median minimizedAffinity: `-7.309`

Template usage in final baseline package:
- `1NRL`: 99 ligands
- `1M13`: 54 ligands
- `3R8D`: 31 ligands

Example selected entries:
- `x00011-1` -> template `3R8D`, pose `x00011-1_3R8D_pose_001`, CNNaffinity `4.969`, minimizedAffinity `-7.103`
- `x00035-1` -> template `1NRL`, pose `x00035-1_1NRL_pose_001`, CNNaffinity `5.090`, minimizedAffinity `-6.555`
- `x00046-1` -> template `1NRL`, pose `x00046-1_1NRL_pose_002`, CNNaffinity `5.525`, minimizedAffinity `-6.025`
- `x00052-1` -> template `1NRL`, pose `x00052-1_1NRL_pose_002`, CNNaffinity `5.616`, minimizedAffinity `-6.521`
- `x00086-1` -> template `1NRL`, pose `x00086-1_1NRL_pose_003`, CNNaffinity `5.124`, minimizedAffinity `-6.013`
- `x00088-1` -> template `3R8D`, pose `x00088-1_3R8D_pose_001`, CNNaffinity `4.910`, minimizedAffinity `-5.802`
- `x00113-1` -> template `1M13`, pose `x00113-1_1M13_pose_003`, CNNaffinity `6.240`, minimizedAffinity `-6.887`
- `x00162-1` -> template `1NRL`, pose `x00162-1_1NRL_pose_001`, CNNaffinity `6.082`, minimizedAffinity `-8.495`
- `x00186-1` -> template `1NRL`, pose `x00186-1_1NRL_pose_003`, CNNaffinity `6.315`, minimizedAffinity `-7.742`
- `x00229-1` -> template `3R8D`, pose `x00229-1_3R8D_pose_001`, CNNaffinity `5.273`, minimizedAffinity `-7.041`
- `x00242-1` -> template `1NRL`, pose `x00242-1_1NRL_pose_001`, CNNaffinity `5.058`, minimizedAffinity `-4.105`
- `x00252-1` -> template `1NRL`, pose `x00252-1_1NRL_pose_002`, CNNaffinity `5.526`, minimizedAffinity `-5.981`

## Validation And Packaging

Baseline exports were subjected to local structural-format validation requiring:

- protein `ATOM` records present
- ligand `HETATM` records present
- terminal `END` record present

Validated PDB files were then bundled into the final baseline submission archive:

- archive name: `baseline_submission_pdb.zip`
- archive members: `184`

## Reproducibility Notes

- the baseline archive contains one monomeric protein-ligand PDB per blinded ligand
- template selection, docking, and pose ranking were applied consistently across the full blinded set
- quantitative values in this report were generated directly from the saved baseline workflow outputs
