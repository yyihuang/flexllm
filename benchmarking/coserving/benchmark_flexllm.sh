set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../../flexflow-serve/build"
source ./set_python_envs.sh

MODEL_NAMES=(
  "meta-llama/Llama-3.1-8B-Instruct"
  # "mistralai/Mistral-Small-24B-Instruct-2501"
  # "meta-llama/Llama-3.1-70B-Instruct"
)
TP_DEGREES=(
  1
  # 2
  # 4
)
ZSIZES=(
  40000
  # 100000
  # 200000
)
TRACES=(
  sharegpt
  # wildchat
)
QPS_vals=(
  5.0
  3.5
  2.0
)
NCPUS=16
FSIZE=76000
CSIZE=2048
MAX_SEQ_LEN=8192
NUM_KV_CACHE_SLOTS=300000
BATCH_SIZE=256
MAX_TOKENS_PER_BATCH=256

MAX_TRAINING_EPOCHS=1000
GRADIENT_ACCUMULATION_STEPS=8
FT_LOGGING_STEPS=100

OUTPUT_FOLDER="../../benchmarking/output/coserving/flexllm"
TRACES_FOLDER="../../benchmarking/traces/burstgpt"
FINETUNING_DATASET="t1"
FINETUNING_DATASET_FILE="${TRACES_FOLDER}/${FINETUNING_DATASET}.json"

mkdir -p $OUTPUT_FOLDER/output
mkdir -p $OUTPUT_FOLDER/logs
mkdir -p $OUTPUT_FOLDER/profiling

for trace in "${TRACES[@]}"; do
  for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME=${MODEL_NAMES[$i]}
    PEFT_MODEL_NAME="${MODEL_NAME}-lora"
    NGPUS=${TP_DEGREES[$i]}
    ZSIZE=${ZSIZES[$i]}
    QPS=${QPS_vals[$i]}
    if [ "$MODEL_NAME" == "meta-llama/Llama-3.1-8B-Instruct" ]; then
      NUM_BWD_LAYER_VALUES=(4 16 32)
      TRACE_FILE="${TRACES_FOLDER}/${trace}_${QPS}_qps.json"
    elif [ "$MODEL_NAME" == "mistralai/Mistral-Small-24B-Instruct-2501" ]; then
      NUM_BWD_LAYER_VALUES=(4 20 40)
      TRACE_FILE="${TRACES_FOLDER}/${trace}_mistral_${QPS}_qps.json"
    elif [ "$MODEL_NAME" == "meta-llama/Llama-3.1-70B-Instruct" ]; then
      NUM_BWD_LAYER_VALUES=(8 40 80)
      TRACE_FILE="${TRACES_FOLDER}/${trace}_${QPS}_qps.json"
    else
      echo "Unknown model: $MODEL_NAME"
      exit 1
    fi
    test -f $TRACE_FILE || { echo "File $TRACE_FILE not found"; exit 1; }
    for NUM_BWD_LAYERS in "${NUM_BWD_LAYER_VALUES[@]}"; do
      OUTPUT_FILE="${OUTPUT_FOLDER}/output/${MODEL_NAME//\//_}_${trace}_bz_${BATCH_SIZE}_tokens_per_batch_${MAX_TOKENS_PER_BATCH}_kv_cache_slots_${NUM_KV_CACHE_SLOTS}_${NUM_BWD_LAYERS}_bwd_layers.json"
      rm $OUTPUT_FILE || true
      
      echo "Running $MODEL_NAME (tp=$NGPUS) on $trace with BZ=$BATCH_SIZE, TOKENS_PER_BATCH=$MAX_TOKENS_PER_BATCH, KV_CACHE_SLOTS=$NUM_KV_CACHE_SLOTS, NUM_BWD_LAYERS=$NUM_BWD_LAYERS"
      
      time ./inference/flexllm/peft_train \
          -ll:cpu $NCPUS -ll:gpu $NGPUS -ll:util $NCPUS \
          -ll:fsize $FSIZE -ll:zsize $ZSIZE -ll:csize $CSIZE \
          -llm-model $MODEL_NAME --fusion \
          -tensor-parallelism-degree $NGPUS \
          -prompt $TRACE_FILE \
          -enable-peft -peft-model $PEFT_MODEL_NAME \
          -finetuning-dataset $FINETUNING_DATASET_FILE \
          --max-training-epochs $MAX_TRAINING_EPOCHS \
          --gradient-accumulation-steps $GRADIENT_ACCUMULATION_STEPS \
          --num-layers-per-finetuning-step $NUM_BWD_LAYERS \
          --num-logging-steps $FT_LOGGING_STEPS \
          -output-file $OUTPUT_FILE \
          -profiling-folder "${OUTPUT_FOLDER}/profiling" \
          --max-requests-per-batch $BATCH_SIZE \
          --max-tokens-per-batch $MAX_TOKENS_PER_BATCH \
          --max-sequence-length $MAX_SEQ_LEN \
          --num-kv-cache-slots $NUM_KV_CACHE_SLOTS \
          --ignore-eos --warmup
      
      sleep 5
    done
  done
done