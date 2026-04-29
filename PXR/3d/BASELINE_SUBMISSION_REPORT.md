# OpenADMET PXR Structure Corrected Submission Report

## Objective

This report documents the corrected full submission package prepared after external blind-test validation identified a subset of failing structures in the original baseline archive. The correction workflow was restricted to the failing compounds and used two targeted remediation strategies: protonation-state repair for chemically invalid cationic states, and GNINA docking fallback for ligands whose UniDock-Pro poses remained highly strained after minimisation.

## Corrected Submission Package

- corrected archive members: `184`
- corrected archive size (bytes): `8354265`
- total blinded ligands in submission: `184`
- corrected ligands replaced in the full archive: `25`
- charge-state corrected ligands: `11`
- GNINA-fallback corrected ligands: `14`

## Root Cause Analysis Of The Failing Subset

Two failure modes were identified among the 25 rejected structures:

1. Chemically invalid protonation states assigned to amide- or lactam-like nitrogens.
2. Catastrophically strained docked poses from UniDock-Pro hybrid docking, especially for ring-rich ligands.

The first class was corrected by rejecting protonated states on nitrogens directly attached to carbonyl-like or sulfonyl-like groups. The second class was corrected by widening the UniDock-Pro pose harvest, applying explicit MMFF94 strain filtering, and invoking GNINA docking when all UniDock-Pro poses for a ligand-template pair remained above the strain threshold.

## Protonation-State Corrections

For the following ligands, the prepared docking state was reverted from an invalid cationic amide/lactam state to the chemically reasonable neutral input state:

- `x00644-1`
- `x00773-1`
- `x02782-1`
- `x02909-1`
- `x02914-1`
- `x03260-1`
- `x03279-1`
- `x03331-1`
- `x03387-1`
- `x03432-1`
- `x03463-1`

Charge-state remediation summary:

- corrected ligands with changed prepared state: `11`
- corrected baseline exports written: `11`

## Strain-Driven GNINA Fallback

For the remaining neutral failures, UniDock-Pro hybrid docking was retained as the first docking engine, but each ligand-template pose set was screened by MMFF94 strain after GNINA minimisation. When all UniDock-Pro poses remained above the strain threshold, GNINA full docking was run against the same receptor/template, followed by GNINA minimisation and CNN rescoring. The lower-strain engine was then retained for baseline pose selection.

Neutral ligands corrected by GNINA fallback:

- `x01016-1`
- `x02715-1`
- `x02776-1`
- `x02797-1`
- `x02828-1`
- `x03037-1`
- `x03096-1`
- `x03152-1`
- `x03223-1`
- `x03234-1`
- `x03282-1`
- `x03319-1`
- `x03400-1`
- `x03401-1`

GNINA fallback summary:

- ligands with fallback-selected pose rows: `14`
- fallback pose rows recorded: `140`
- corrected neutral baseline exports written: `14`

Example GNINA-fallback selections:
- `x01016-1` -> template `3R8D`, pose `x01016-1_3R8D_gninafb_pose_003`, CNNaffinity `5.482`, minimizedAffinity `-5.867`
- `x02715-1` -> template `1M13`, pose `x02715-1_1M13_gninafb_pose_001`, CNNaffinity `6.911`, minimizedAffinity `-7.776`
- `x02776-1` -> template `1NRL`, pose `x02776-1_1NRL_gninafb_pose_001`, CNNaffinity `6.687`, minimizedAffinity `-9.578`
- `x02797-1` -> template `1NRL`, pose `x02797-1_1NRL_gninafb_pose_002`, CNNaffinity `7.193`, minimizedAffinity `-8.989`
- `x02828-1` -> template `1NRL`, pose `x02828-1_1NRL_gninafb_pose_002`, CNNaffinity `6.384`, minimizedAffinity `-9.034`
- `x03037-1` -> template `1NRL`, pose `x03037-1_1NRL_gninafb_pose_004`, CNNaffinity `6.486`, minimizedAffinity `-8.265`
- `x03096-1` -> template `1M13`, pose `x03096-1_1M13_gninafb_pose_005`, CNNaffinity `6.910`, minimizedAffinity `-8.416`
- `x03152-1` -> template `1NRL`, pose `x03152-1_1NRL_gninafb_pose_001`, CNNaffinity `6.213`, minimizedAffinity `-9.506`

## Final Selection And Packaging

The corrected full archive was assembled by overlaying the 25 corrected ligand-protein PDB files onto the original 184-member baseline archive. All corrected PDBs passed the local structural-format checks used in the workflow before packaging.

- replaced PDB entries in the full archive: `25`
- retained original baseline entries: `159`
- final corrected archive name: `submission_pdb.zip`

## Practical Interpretation

The corrected archive differs from the original baseline submission only for the 25 compounds that failed external validation. Eleven were repaired by chemistry-aware protonation filtering, and fourteen were repaired by switching from strained UniDock-Pro poses to GNINA-docked alternatives selected after minimisation and strain screening. This corrected full archive is therefore the recommended replacement package for resubmission.
