import os,sys,time
from scipy.stats import pearsonr
import pandas as pd

# python MPNN_public.py ADME_HLM_train.csv ADME_HLM_test.csv default
# mode == default, hyperopt

trainset = sys.argv[1]
testset = sys.argv[2]
mode = sys.argv[3]



if mode == 'Opt':
      
    # 4. MPNN2-OPT
    os.system('chemprop predict --test-path %s --smiles-columns smiles --molecule-featurizers v1_rdkit_2d_normalized --no-descriptor-scaling --model-path %s_hp_rdkit_checkpoints --preds-path %s_hp_oad_rdkit_preds.csv' %(testset,sys.argv[1][:-4],sys.argv[2][0:-10]))  
