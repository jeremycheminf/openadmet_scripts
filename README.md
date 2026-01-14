final_corrected_analysis_with_predictions_v2.py is a script to test gline and kate-coder-pro in VScode as agents to find the best model for HLM and RLM properties.
They were guided with info on RDKIT and asked to look for ML available as well as as some extra features it could look at. There was some need to ask more to get some performance on internal test set (recent compounds from training), but results out of the box were good and impressive at it went by itself to read training files, find python libraries, run the code, and give results.
Better results can be done with extra tuning, datasets, but for a run, go for lunch, back it can give some ideas, and I guess one can ask for plots as well.

Full submissions for other endpoints are
- LogD: chemprop with hyper parameter tuning with external datasets - MPNN_public_oad_model_train.py; datasets are from https://github.com/myzhengSIMM/RTlogD/tree/main with training (original_data) and test set (chembl32); plus AZ logD data (TDC); and https://github.com/nanxstats/logd74 
- all others, chemprop with chemeleon single task using all training data and predicting test sets; this code can also be used for HLM Clint and RLM Clint; performances were similar to the agent looking for best models, but it's easier to run chemeleon.
