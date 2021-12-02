# This shell script executes the training and testing.  It then saves the output.
echo "Starting pipeline..."

# stop the execution of the script if the pipeline has an error
# "unofficial bash strict mode"
set -euo pipefail
IFS=$'\n\t'

# Specify dataset, first argument
DATASET=${1:-1}

# Directory to save outputs for each run
OUTPUT_DIR=results/${DATASET}
RNG_SEED=$RANDOM
OUTPUT_DIR+="_${RNG_SEED}"

mkdir -p ${OUTPUT_DIR}

# copy both standard output and standard error streams to file while still 
# being visible in the terminal.  Append if file already exists
python train.py \
DATASET ${DATASET} \
OUTPUT_DIR ${OUTPUT_DIR} \
2>&1 | tee -a ${OUTPUT_DIR}/log.txt
