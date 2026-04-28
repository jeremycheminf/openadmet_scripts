import os,sys,time
from scipy.stats import pearsonr
import pandas as pd

# python MPNN_public.py ADME_HLM_train.csv ADME_HLM_test.csv default
# mode == default, hyperopt

trainset = sys.argv[1]
testset = sys.argv[2]
mode = sys.argv[3]



if mode == 'Opt':
      
    os.system('chemprop hpopt --data-path %s --task-type regression --metric r2 --raytune-use-gpu --raytune-num-samples 10 --molecule-featurizers v1_rdkit_2d_normalized --no-descriptor-scaling --hpopt-save-dir %s_hp_rdkit' %(trainset,sys.argv[1][:-4]))
    ## train and prediction  
    os.system('chemprop train --data-path %s --task-type regression --molecule-featurizers v1_rdkit_2d_normalized --no-descriptor-scaling --metric r2 --config-path %s_hp_rdkit/best_config.toml --output-dir %s_hp_rdkit_checkpoints' %(trainset,sys.argv[1][:-4], sys.argv[1][:-4]))

