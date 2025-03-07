set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../flexflow-serve/build"

reset
# make -j 
source ./set_python_envs.sh

MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"
NGPUS=4
NCPUS=16
FSIZE=76000
ZSIZE=200000

OUTPUT_FOLDER="../../benchmarking/output/incr_decoding"
TRACES_FOLDER="../../benchmarking/traces"
MAX_SEQ_LEN=8192
NUM_KV_CACHE_SLOTS=$((MAX_SEQ_LEN * 4))
batch_sizes=(
    128
    64
    32
    16
    8
    4
)

trace_files=(
    sharegpt
    wildchat
)

max_tokens_per_batch_values=(
    2048
    1024
    512
    256
    128
)

mkdir -p $OUTPUT_FOLDER/output
mkdir -p $OUTPUT_FOLDER/logs
mkdir -p $OUTPUT_FOLDER/profiling 

for i in "${!trace_files[@]}"; do
for k in "${!batch_sizes[@]}"; do
for j in "${!max_tokens_per_batch_values[@]}"; do
    
    TRACE_FILE="${TRACES_FOLDER}/${trace_files[$i]}.json"
    test -f $TRACE_FILE || { echo "File $TRACE_FILE not found"; exit 1; }
    MAX_TOKENS_PER_BATCH=${max_tokens_per_batch_values[$j]}
    BATCH_SIZE=${batch_sizes[$k]}
    echo "Running $TRACE_FILE with BZ=$BATCH_SIZE, TOKENS_PER_BATCH=$MAX_TOKENS_PER_BATCH, KV_CACHE_SLOTS=$NUM_KV_CACHE_SLOTS"

    OUTPUT_FILE="${OUTPUT_FOLDER}/output/incr_dec_${trace_files[$i]}_bz_${MAX_TOKENS_PER_BATCH}_tokens_per_batch_${MAX_TOKENS_PER_BATCH}_kv_cache_slots_${NUM_KV_CACHE_SLOTS}.json"
    LOG_FILE="${OUTPUT_FOLDER}/logs/incr_dec_${trace_files[$i]}_bz_${MAX_TOKENS_PER_BATCH}_tokens_per_batch_${MAX_TOKENS_PER_BATCH}_kv_cache_slots_${NUM_KV_CACHE_SLOTS}.out"
    rm $OUTPUT_FILE $LOG_FILE || true
    
    time ./inference/incr_decoding/incr_decoding \
        -ll:cpu $NCPUS -ll:gpu $NGPUS -ll:util $NCPUS \
        -ll:fsize $FSIZE -ll:zsize $ZSIZE \
        -llm-model $MODEL_NAME --fusion \
        -tensor-parallelism-degree $NGPUS \
        -prompt $TRACE_FILE \
        -output-file $OUTPUT_FILE \
        -profiling-folder "${OUTPUT_FOLDER}/profiling" \
        --max-requests-per-batch $BATCH_SIZE \
        --max-tokens-per-batch $MAX_TOKENS_PER_BATCH \
        --max-sequence-length $MAX_SEQ_LEN \
        --num-kv-cache-slots $NUM_KV_CACHE_SLOTS \
        --warmup 2>&1 | tee $LOG_FILE
done
done
done

