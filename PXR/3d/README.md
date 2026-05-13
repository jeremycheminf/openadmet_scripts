# OpenADMET PXR 3D Structure Pipeline

This workspace contains a restartable structure-submission pipeline for the OpenADMET PXR challenge.

## Campaign Summary

This workspace evolved from a single restartable docking pipeline into a set of structure-prediction campaign arms for the PXR blind structure track.

The main docking-based rounds completed here were:

- `GNINA` baseline docking and minimization
- `UniDock` baseline rounds, including no-protonation and rerefined-template variants
- `UniDock-Pro` hybrid docking rounds, including broader template coverage and strain-based filtering
- `rDock` rounds, ending in a rerefined-template top-3 shortlist workflow using scaffold/MCS matching plus RDKit shape preselection

In practical leaderboard terms, the outcome was poor across the docking families that were tested. The submissions from `GNINA`, `UniDock`, `UniDock-Pro`, and `rDock` all remained low in the ranking rather than becoming competitive structure-track solutions. The different engines and template-selection policies changed details of the outputs, but they did not produce a strong-performing submission.

Because of that, this workspace should be read mainly as a record of the methods tested, the restartable infrastructure built around them, and the template/docking experiments that were completed, rather than as a successful final modeling solution.

## Layout

- `data/`
  Runtime inputs, downloaded challenge CSVs, curated template assets, and intermediate ligand files.
- `scripts/`
  Restartable stage scripts and the main orchestration CLI.
- `results/`
  Manifests, stage state, scored poses, final PDBs, submission zip, and the PDF report.
- `PLAN.md`
  Saved implementation plan for this workspace.

## Main Entry Point

Run from the repo root:

```powershell
python OpenADMET\PXR\3D\scripts\run_structure_pipeline.py `
  --challenge-csv OpenADMET\PXR\3D\data\pxr_structure_test.csv `
  --output-dir OpenADMET\PXR\3D\results `
  --resume
```

Useful flags:

- `--resume`
  Skip completed work based on manifests and `_SUCCESS` markers.
- `--skip-baseline-package`
  Skip the fast GNINA baseline package and go straight to the slower MD-rescored selection path.
- `--baseline-only`
  Stop after creating the fast GNINA baseline package, without launching MM-GBSA/OpenBPMD refinement.
- `--strain-threshold <float>`
  Maximum allowed MMFF94 strain energy in kcal/mol for baseline pose retention after GNINA minimization. Default: `10.0`.
- `--strain-num-conf <int>`
  Number of conformers used when estimating MMFF94 strain relative to the approximate minimum. Default: `30`.
- `--force-stage <stage>`
  Re-run a stage even if it looks complete.
- `--force-ligand <ligand_id>`
  Re-run work for one ligand during docking, scoring, export, or validation.
- `--tutorial-repo <path>`
  Optional local clone of the official tutorial/validation repo.
- `--openmm-url <url>`
  OpenMM microservice base URL for OpenBPMD/MM-GBSA.
- `--protonation-ph <float>`
  Target pH for ligand state preparation before docking. Default: `7.4`.
- `--max-microstates <int>`
  Maximum protonation states to record per ligand in the assignment manifest.
- `--protonation-fast`
  Use the fast Dimorphite-only path instead of pkasolver-ranked state preparation.

## Stages

1. `template_panel`
2. `assign_templates`
3. `run_docking_and_rescoring`
   UniDock-Pro hybrid docking, GNINA minimization/CNN rescoring, then MMFF94 strain filtering with a one-pose rescue fallback if all poses exceed the strain threshold.
4. `baseline_submission`
5. `select_final_pose`
6. `export_submission`
7. `validate_submission`
8. `package_submission`
9. `generate_report`

Each expensive stage writes machine-readable manifests and resumes from those manifests on rerun.

## Outputs

- `results/template_manifest.csv`
- `results/assignment_manifest.csv`
- `results/pose_manifest.csv`
- `results/submission_manifest.csv`
- `results/baseline_submission_pdb/*.pdb`
- `results/baseline_submission_pdb.zip`
- `results/submission_pdb/*.pdb`
- `results/submission_pdb.zip`
- `results/PXR_structure_report.pdf`
- `results/state.json`

## Notes

- This pipeline prepares a local submission zip only. It does not perform API submission.
- Final PDBs are exported as monomeric protein + ligand complexes, one file per ligand.
- The pipeline reuses the existing local docking/OpenMM stack in `docker-dockshape-project-claude` and `docker-openmmmin`.
- For 2D ligand inputs, the assignment stage now records both the input SMILES and the prepared docking SMILES, along with their formal charges and the ranked protonation states used at the chosen pH.
- By default, the pipeline now creates a fast baseline submission package from GNINA-minimized poses, filtering out high-strain poses before picking the baseline winner by GNINA CNN score.
- Later campaign arms broadened beyond the original default pipeline and included alternate UniDock, UniDock-Pro, and `rDock` workflows using the rerefined PXR template panel and more aggressive template-routing logic.
- Despite those additional rounds, the docking-based submissions stayed near the bottom of the visible leaderboard and should be considered exploratory rather than successful.

## OpenFold With Rerefined Templates

The OpenFold cofolding workflow can now use local rerefined PXR complexes instead of falling back to the original RCSB mmCIF structures.

Build a compatible template manifest from a local rerefined structure folder:

```powershell
python OpenADMET\PXR\3D\scripts\build_rerefined_template_manifest.py `
  --input-dir <local_rerefined_structure_dir> `
  --output-dir OpenADMET\PXR\3D\results\rerefined_template_panel
```

Then run OpenFold against that manifest:

```powershell
python OpenADMET\PXR\3D\scripts\run_openfold3_cofold.py `
  --template-manifest OpenADMET\PXR\3D\results\rerefined_template_panel\template_panel.csv `
  --output-dir OpenADMET\PXR\3D\results\openfold3_cofold_rerefined
```

If the rerefined template folder has extra metadata such as explicit `chain_id`, `ligand_code`, or `resolution`, pass it through `--metadata-csv` when building the manifest.
