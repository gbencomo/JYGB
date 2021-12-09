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

LEARNING_RATE=1e-2
EPOCHS=200
BATCH_SIZE=16
[ "$DATASET" == "mbm" ] && BATCH_SIZE=15
AUGMENT=1
FILTERS=64
CONV=2
WEIGHT_DECAY=1e-3
MOMENTUM=0.9
VAL_PERCENT=0.2

# train model with above conditions; save output
python train.py \
-d ${DATASET} \
-lr ${LEARNING_RATE} \
-e ${EPOCHS} \
-b ${BATCH_SIZE} \
-a ${AUGMENT} \
-uf ${FILTERS} \
-c ${CONV} \
-wd ${WEIGHT_DECAY} \
-m ${MOMENTUM} \
-s ${RNG_SEED} \
-v ${VAL_PERCENT} \
-sp ${OUTPUT_DIR} \ 
--plot \
2>&1 | tee -a ${OUTPUT_DIR}/log.txt
