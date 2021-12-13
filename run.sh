# stop the execution of the script if the pipeline has an error
# "unofficial bash strict mode"
set -euo pipefail
IFS=$'\n\t'
set -e

# Specify dataset, first argument
DATASET=${1:-1}

# Directory to save outputs for each run
OUTPUT_DIR=results/${DATASET}
RNG_SEED=$RANDOM
OUTPUT_DIR+="_${RNG_SEED}"

LEARNING_RATE=0.001
EPOCHS=300
BATCH_SIZE=16
[ "$DATASET" == "mbm" ] && BATCH_SIZE=15
FILTERS=64
CONV=2
WEIGHT_DECAY=0.001
MOMENTUM=0.9

mkdir -p ${OUTPUT_DIR}

# train model with above conditions; save output
python train.py \
-d ${DATASET} \
-sp ${OUTPUT_DIR} \
-lr ${LEARNING_RATE} \
-e ${EPOCHS} \
-b ${BATCH_SIZE} \
-uf ${FILTERS} \
-c ${CONV} \
-wd ${WEIGHT_DECAY} \
-m ${MOMENTUM} \
-s ${RNG_SEED} \
--plot \
#2>&1 | tee -a ${OUTPUT_DIR}/log.txt
