set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../flexflow-serve/build"

# reset
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

python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id=\"${MODEL_NAME}\", allow_patterns=\"*.safetensors\", max_workers=30)"
python ../inference/utils/download_hf_model.py --half-precision-only $MODEL_NAME

for i in "${!trace_files[@]}"; do
for k in "${!batch_sizes[@]}"; do
for j in "${!max_tokens_per_batch_values[@]}"; do
    
    TRACE_FILE="${TRACES_FOLDER}/${trace_files[$i]}.json"
    test -f $TRACE_FILE || { echo "File $TRACE_FILE not found"; exit 1; }
    MAX_TOKENS_PER_BATCH=${max_tokens_per_batch_values[$j]}
    BATCH_SIZE=${batch_sizes[$k]}
    echo "Running $TRACE_FILE with BZ=$BATCH_SIZE, TOKENS_PER_BATCH=$MAX_TOKENS_PER_BATCH, KV_CACHE_SLOTS=$NUM_KV_CACHE_SLOTS"

    OUTPUT_FILE="${OUTPUT_FOLDER}/output/${trace_files[$i]}_bz_${MAX_TOKENS_PER_BATCH}_tokens_per_batch_${MAX_TOKENS_PER_BATCH}_kv_cache_slots_${NUM_KV_CACHE_SLOTS}.json"
    LOG_FILE="${OUTPUT_FOLDER}/logs/${trace_files[$i]}_bz_${MAX_TOKENS_PER_BATCH}_tokens_per_batch_${MAX_TOKENS_PER_BATCH}_kv_cache_slots_${NUM_KV_CACHE_SLOTS}.out"
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

benchmark_vllm() {
    python3 -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --tensor-parallel-size 1 \
        --swap-space 16 \
        --disable-log-stats \
        --disable-log-requests\
        --load-format dummy
    python3 benchmark_serving.py \
        --save-result \
        --result-dir results/ \
        --result-filename serving_llama8B_tp1_sharegpt_qps_4.json \
        --request-rate 4\
        --model meta-llama/Meta-Llama-3.1-8B-Instruct 
        --backend vllm 
        --dataset-name sharegpt 
        --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json 
        --num-prompts 200
}