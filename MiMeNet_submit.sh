##example batch script

#!/bin/bash
#SBATCH --job-name=MiMeNet
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100pcie
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=96000
#SBATCH --error=log/MiMeNet_%j.err
#SBATCH --output=log/MiMeNet-%j.out


# Activate conda environment
module load conda
nvidia-smi

conda activate MiMeNet

INPUT_DIR=$1
STUDY=$2

# Run training
# note: metabolite annotation + labels don't work together with external data
# Define output prefix
#i.e. Camilleri, iHMP, PRISM, PROTECT, 1000IBD
OUTPUT="${STUDY}_10_CLR_fin"

# Run pipeline example
#python -u /MiMeNet_train_2.py \
#    -labels "${INPUT_DIR}/data/${STUDY}/${STUDY}_meta.csv" \
#    -annotation "${INPUT_DIR}/data/${STUDY}/${STUDY}_annotation.csv" \
#    -micro_norm CLR \
#    -metab_norm CLR \
#    -num_run_cv 10 \
#    -output "$OUTPUT" \
#    -num_run 10 \
#    -num_cv 10 \
#    -micro "${INPUT_DIR}/data/${STUDY}/${STUDY}_shortbred_counts.csv" \
#    -metab "${INPUT_DIR}/data/${STUDY}/${STUDY}_mtb.csv" \
#    -num_background 100
    