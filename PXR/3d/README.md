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

Tested with and without protonation with dimorphite_dl but no improvement, or was even worst. Some issues with unidock-pro in hybrid mode and compounds failing after docking with wrong bond orders and high strain.
Posebuster strain energy style (MMFF) was used to filter poses, but no impact on leaderboard.
