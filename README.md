final_corrected_analysis_with_predictions_v2.py is a script to test gline and kate-coder-pro in VScode as agents to find the best model for HLM and RLM properties.
They were guided with info on RDKIT and asked to look for ML available as well as as some extra features it could look at. There was some need to ask more to get some performance on internal test set (recent compounds from training), but results out of the box were good and impressive at it went by itself to read training files, find python libraries, run the code, and give results.
Better results can be done with extra tuning, datasets, but for a run, go for lunch, back it can give some ideas, and I guess one can ask for plots as well.
R2 for HLM and MLM were around 0.3 and 0.35, it did better on internal validation than leaderboard and could be from overfitting.

Full submissions for other endpoints are
- LogD: chemprop with hyper parameter tuning with external datasets - MPNN_public_oad_model_train.py; datasets are from https://github.com/myzhengSIMM/RTlogD/tree/main with training (original_data) and test set (chembl32); plus AZ logD data (TDC); and https://github.com/nanxstats/logd74 
- all others, chemprop with chemeleon single task using all training data and predicting test sets; this code can also be used for HLM Clint and RLM Clint; performances were similar to the agent looking for best models, but it's easier to run chemeleon.

Tested with little effect, removing first 100/200 compounds by compound ID to see if older compounds had effect on predictions. I didn't check similarity to test set to keep in blinded as it would be in a project.
On the agent for creating scripts, issues that came up quite often were:
- Warning on RDKit and fingerprint generator, not a big problem but the agent reads the entire output, so need to hide them otherwise too many tokens used
- Chemprop not easy to use and it reverts mostly to v1 even if I tell I have v2
- RDKit descriptors end up usually with hardcoded small list (MW, LogP, HBD, HBA, TPSA) rather than the full list
- If asked about new descriptors it doesn't know, it reverts to RDKit small list but creates a function with the name asked. For example, jazzy_gen would actually use RDkit small list from the agent rather than jazzy
- it can create lots of files with readme, markdown, pictures... even when not asked about it, sometimes good, other time too much
- - over interpret its own results with R2, or invent results saying "done" and good while the code didn't work, and ends in a loop

The last submission I did was using even more descriptors such as jazzy,  avalon and qm from a chemprop model trained on QM9 data, and the logD model from the logD endpoint. But that was for HLM, MLM, KSol and CACO2 only.
