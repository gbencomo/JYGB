# This shell script executes the training and testing.  It then saves the output.
echo "Starting pipeline..."

# stop the execution of the script if the pipeline has an error
# "unofficial bash strict mode"
set -euo pipefail
IFS=$'\n\t'

DATASET=${1:-1}

OUTPUT_DIR=results/${DATASET}
RNG_SEED=$RANDOM
OUTPUT_DIR+="_${RNG_SEED}"

mkdir -p ${OUTPUT_DIR}

