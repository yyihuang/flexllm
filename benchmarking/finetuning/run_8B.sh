set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../../flexflow-serve/build"

source ./set_python_envs.sh

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
NGPUS=1
NCPUS=16
FSIZE=76000
ZSIZE=40000
CSIZE=2048
MAX_SEQ_LEN=8192
MAX_REQUESTS_PER_BATCH=1
MAX_TOKENS_PER_BATCH=8192
NUM_KV_CACHE_SLOTS=8192
MAX_TRAINING_EPOCHS=1

OUTPUT_FOLDER="../../benchmarking/output/finetuning/8B"
TRACES_FOLDER="../../benchmarking/traces"
FINETUNING_DATASET="t1"
FINETUNING_DATASET_FILE="${TRACES_FOLDER}/${FINETUNING_DATASET}.json"
PEFT_MODEL_NAME="goliaro/llama-3.1-8b-lora-throughput-test"
OUTPUT_FILE="${OUTPUT_FOLDER}/output/${FINETUNING_DATASET}_${MODEL_NAME}_finetuning.json"

mkdir -p $OUTPUT_FOLDER/output
mkdir -p $OUTPUT_FOLDER/logs
mkdir -p $OUTPUT_FOLDER/profiling 

./inference/flexllm/peft_train \
    -ll:cpu $NCPUS -ll:gpu $NGPUS -ll:util $NCPUS \
    -ll:fsize $FSIZE -ll:zsize $ZSIZE -ll:csize $CSIZE \
    -llm-model $MODEL_NAME --fusion \
    -tensor-parallelism-degree $NGPUS \
    -enable-peft -peft-model $PEFT_MODEL_NAME \
    -finetuning-dataset $FINETUNING_DATASET_FILE \
    --max-training-epochs $MAX_TRAINING_EPOCHS \
    --gradient-accumulation-steps 8 \
    --num-logging-steps 1 \
    --max-finetuning-samples 160 \
    -output-file $OUTPUT_FILE \
    -profiling-folder "${OUTPUT_FOLDER}/profiling" \
    --max-requests-per-batch $MAX_REQUESTS_PER_BATCH \
    --max-tokens-per-batch $MAX_TOKENS_PER_BATCH \
    --max-sequence-length $MAX_SEQ_LEN \
    --num-kv-cache-slots $NUM_KV_CACHE_SLOTS \
    --warmup 2>&1 | tee $LOG_FILE