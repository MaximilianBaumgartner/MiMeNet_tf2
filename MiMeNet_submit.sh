#!/bin/bash
#SBATCH --job-name=MiMeNet
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l4_gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=32000
#SBATCH --error=log/MiMeNet_%j.err
#SBATCH --output=log/MiMeNet-%j.out


# Activate environment
module load conda
module load CUDA/12.3.0
module load cuDNN/8.9.2.26-CUDA-12.3.0

source $CONDA_BASH
conda activate MiMeNet

# Add MiMeNet to Python path
export PYTHONPATH=/nobackup/lab_campbell/Max/conda/pkgs/MiMeNet/MiMeNet/src:$PYTHONPATH


INPUT_DIR=$1

# Run training
python -u /nobackup/lab_campbell/Max/conda/pkgs/MiMeNet/MiMeNet/MiMeNet_train.py \
-micro $INPUT_DIR/data/IBD/microbiome_PRISM.csv \
-metab $INPUT_DIR/data/IBD/metabolome_PRISM.csv \
-external_micro $INPUT_DIR/data/IBD/microbiome_external.csv \
-external_metab $INPUT_DIR/data/IBD/metabolome_external.csv \
-micro_norm None \
-metab_norm CLR \
-net_params $INPUT_DIR/results/IBD/network_parameters.txt \
-labels $INPUT_DIR/data/IBD/diagnosis_PRISM.csv \
-num_run_cv 1 \
-num_background 20 \
-output IBD \
-background $INPUT_DIR/results/IBD/BG \
-num_run 2
