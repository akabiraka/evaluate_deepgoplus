#!/usr/bin/sh

#SBATCH --job-name=GO
#SBATCH --output=/scratch/akabir4/deepgoplus/ak_outputs/argo_logs/argo-%j.out
#SBATCH --error=/scratch/akabir4/deepgoplus/ak_outputs/argo_logs/argo-%j.err
#SBATCH --mail-user=<akabir4@gmu.edu>
#SBATCH --mail-type=BEGIN,END,FAIL

##--------------CPU jobs------------------
##SBATCH --partition=all-LoPri
##SBATCH --cpus-per-task=4
##SBATCH --mem=32000MB

##python data_preprocess/compute_GO_terms_topo_matrix.py.py
##python data_preprocess/compute_seq_rep_using_esm1b.py


##--------------CPU array jobs------------------
##SBATCH --partition=all-LoPri
##SBATCH --cpus-per-task=4
##SBATCH --mem=16000MB
##SBATCH --array=0-2

##python data_preprocess/expand_dev_test_set.py


##--------------GPU jobs------------------
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --mem=32G                               # for CPU mem allocation
##SBATCH --nodes=1                            # Request N nodes
##SBATCH --ntasks-per-node=8                  # Request n   cores per node


##nvidia-smi
python ak_codes/train_val_test.py

##python models/test.py
##python models/eval_pred_scores.py

##python models/example_esm_1b.py
