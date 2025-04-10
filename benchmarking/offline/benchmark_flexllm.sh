set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../../flexflow-serve/build"
source ./set_python_envs.sh

MODEL_NAMES=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "mistralai/Mistral-Small-24B-Instruct-2501"
  "meta-llama/Llama-3.1-70B-Instruct"
)
TP_DEGREES=(
  1
  2
  4
)
TRACES=(
  sharegpt
  # wildchat
)
NCPUS=16
FSIZE=76000
ZSIZE=200000
CSIZE=2048
MAX_SEQ_LEN=8192
NUM_KV_CACHE_SLOTS=300000
BATCH_SIZE=256
MAX_TOKENS_PER_BATCH=256

OUTPUT_FOLDER="../../benchmarking/output/offline/flexllm"
TRACES_FOLDER="../../benchmarking/traces"


mkdir -p $OUTPUT_FOLDER/output
mkdir -p $OUTPUT_FOLDER/logs
mkdir -p $OUTPUT_FOLDER/profiling

for trace in "${TRACES[@]}"; do
    
    for i in "${!MODEL_NAMES[@]}"; do
        MODEL_NAME=${MODEL_NAMES[$i]}
        NGPUS=${TP_DEGREES[$i]}

        if [ "$MODEL_NAME" == "mistralai/Mistral-Small-24B-Instruct-2501" ]; then
            TRACE_FILE="${TRACES_FOLDER}/${trace}_mistral.json"
        else
            TRACE_FILE="${TRACES_FOLDER}/${trace}.json"
        fi
        test -f $TRACE_FILE || { echo "File $TRACE_FILE not found"; exit 1; }
        
        OUTPUT_FILE="${OUTPUT_FOLDER}/output/${MODEL_NAME//\//_}_${trace}_bz_${BATCH_SIZE}_tokens_per_batch_${MAX_TOKENS_PER_BATCH}_kv_cache_slots_${NUM_KV_CACHE_SLOTS}.json"
        rm $OUTPUT_FILE || true
        
        echo "Running $MODEL_NAME (tp=$NGPUS) on $trace with BZ=$BATCH_SIZE, TOKENS_PER_BATCH=$MAX_TOKENS_PER_BATCH, KV_CACHE_SLOTS=$NUM_KV_CACHE_SLOTS"
        
        time ./inference/incr_decoding/incr_decoding \
            -ll:cpu $NCPUS -ll:gpu $NGPUS -ll:util $NCPUS \
            -ll:fsize $FSIZE -ll:zsize $ZSIZE -ll:csize $CSIZE \
            -llm-model $MODEL_NAME --fusion \
            -tensor-parallelism-degree $NGPUS \
            -prompt $TRACE_FILE \
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