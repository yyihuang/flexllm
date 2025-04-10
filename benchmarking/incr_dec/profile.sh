set -x
# set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../../flexflow-serve/build"

# reset
# make -j 
source ./set_python_envs.sh

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
NGPUS=1
NCPUS=16
FSIZE=76000
ZSIZE=40000
CSIZE=2048

OUTPUT_FOLDER="$PSCRATCH/benchmarking/output/nsys/8B"
TRACES_FOLDER="../../benchmarking/traces"
MAX_SEQ_LEN=8192
NUM_KV_CACHE_SLOTS=240000
batch_size=256
max_tokens_per_batch=128
trace_file=sharegpt


mkdir -p $OUTPUT_FOLDER/output
mkdir -p $OUTPUT_FOLDER/logs
mkdir -p $OUTPUT_FOLDER/profiling 

# python -c "from huggingface_hub import snapshot_download; \
#     snapshot_download(repo_id=\"${MODEL_NAME}\", allow_patterns=\"*.safetensors\", max_workers=30)"
# python ../inference/utils/download_hf_model.py --half-precision-only $MODEL_NAME

    
TRACE_FILE="${TRACES_FOLDER}/${trace_file}.json"
test -f $TRACE_FILE || { echo "File $TRACE_FILE not found"; exit 1; }
echo "Running $TRACE_FILE with BZ=$batch_size, TOKENS_PER_BATCH=$max_tokens_per_batch, KV_CACHE_SLOTS=$NUM_KV_CACHE_SLOTS"

OUTPUT_FILE="${OUTPUT_FOLDER}/output/${trace_file}_bz_${batch_size}_tokens_per_batch_${max_tokens_per_batch}_kv_cache_slots_${NUM_KV_CACHE_SLOTS}.json"
LOG_FILE="${OUTPUT_FOLDER}/logs/${trace_file}_bz_${batch_size}_tokens_per_batch_${max_tokens_per_batch}_kv_cache_slots_${NUM_KV_CACHE_SLOTS}.out"
rm $OUTPUT_FILE $LOG_FILE || true

# export LEGION_BACKTRACE=1
nsys profile \
--output ${OUTPUT_FOLDER}/report.nsys-rep \
--force-overwrite true \
--stats true \
--sample none \
--trace cuda \
./inference/incr_decoding/incr_decoding \
    -ll:cpu $NCPUS -ll:gpu $NGPUS -ll:util $NCPUS \
    -ll:fsize $FSIZE -ll:zsize $ZSIZE -ll:csize $CSIZE \
    -llm-model $MODEL_NAME --fusion \
    -tensor-parallelism-degree $NGPUS \
    -prompt $TRACE_FILE \
    -output-file $OUTPUT_FILE \
    -profiling-folder "${OUTPUT_FOLDER}/profiling" \
    --max-requests-per-batch $batch_size \
    --max-tokens-per-batch $max_tokens_per_batch \
    --max-sequence-length $MAX_SEQ_LEN \
    --num-kv-cache-slots $NUM_KV_CACHE_SLOTS \
    --ignore-eos \
    --warmup
