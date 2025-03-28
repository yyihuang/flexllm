#!/bin/bash
#SBATCH --account=m4138
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --constraint gpu,ss11,a100,hbm80g
#SBATCH --time=01:00:00
#SBATCH --job-name=incr_dec70b
#SBATCH --output=/global/homes/g/goliaro/flexllm/benchmarking/output/slurm/%x_%A_%a.out
#SBATCH --error=/global/homes/g/goliaro/flexllm/benchmarking/output/slurm/%x_%A_%a.err
#SBATCH --array=0-69

set -x
set -e

# Change directory to the script’s location relative to the build directory
CWD_="/global/homes/g/goliaro/flexllm"
cd "$CWD_/flexflow-serve/build"

# Set up the environment
source ./set_python_envs.sh

# Define constant variables
MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"
NGPUS=4
NCPUS=16
FSIZE=76000
ZSIZE=200000
CSIZE=2048

OUTPUT_FOLDER="../../benchmarking/output/incr_decoding/70B"
TRACES_FOLDER="../../benchmarking/traces"
MAX_SEQ_LEN=8192
NUM_KV_CACHE_SLOTS=240000

# Define arrays corresponding to the original for-loop dimensions
batch_sizes=(256 128 64 32 16 8 4)
trace_files=(sharegpt wildchat)
max_tokens_per_batch_values=(2048 1024 512 256 128)

# Create necessary output directories
mkdir -p "$OUTPUT_FOLDER/output"
mkdir -p "$OUTPUT_FOLDER/logs"
mkdir -p "$OUTPUT_FOLDER/profiling"

# Convert the SLURM_ARRAY_TASK_ID (0 to 69) to the three indices:
# Total combinations = 2 * 7 * 5 = 70. For a given TASK_ID:
#   i = TASK_ID / 35         (index for trace_files)
#   rem = TASK_ID % 35
#   k = rem / 5              (index for batch_sizes)
#   j = rem % 5              (index for max_tokens_per_batch_values)
i=$(( SLURM_ARRAY_TASK_ID / 35 ))
rem=$(( SLURM_ARRAY_TASK_ID % 35 ))
k=$(( rem / 5 ))
j=$(( rem % 5 ))

TRACE_FILE="${TRACES_FOLDER}/${trace_files[$i]}.json"
if [ ! -f "$TRACE_FILE" ]; then
    echo "File $TRACE_FILE not found"
    exit 1
fi

MAX_TOKENS_PER_BATCH=${max_tokens_per_batch_values[$j]}
BATCH_SIZE=${batch_sizes[$k]}

echo "Running $TRACE_FILE with BZ=$BATCH_SIZE, TOKENS_PER_BATCH=$MAX_TOKENS_PER_BATCH, KV_CACHE_SLOTS=$NUM_KV_CACHE_SLOTS"

OUTPUT_FILE="${OUTPUT_FOLDER}/output/${trace_files[$i]}_bz_${BATCH_SIZE}_tokens_per_batch_${MAX_TOKENS_PER_BATCH}_kv_cache_slots_${NUM_KV_CACHE_SLOTS}.json"
LOG_FILE="${OUTPUT_FOLDER}/logs/${trace_files[$i]}_bz_${BATCH_SIZE}_tokens_per_batch_${MAX_TOKENS_PER_BATCH}_kv_cache_slots_${NUM_KV_CACHE_SLOTS}.out"
rm -f "$OUTPUT_FILE" "$LOG_FILE"

# Execute the inference command
time ./inference/incr_decoding/incr_decoding \
    -ll:cpu $NCPUS -ll:gpu $NGPUS -ll:util $NCPUS \
    -ll:fsize $FSIZE -ll:zsize $ZSIZE -ll:csize $CSIZE \
    -llm-model $MODEL_NAME --fusion \
    -tensor-parallelism-degree $NGPUS \
    -prompt "$TRACE_FILE" \
    -output-file "$OUTPUT_FILE" \
    -profiling-folder "${OUTPUT_FOLDER}/profiling" \
    --max-requests-per-batch $BATCH_SIZE \
    --max-tokens-per-batch $MAX_TOKENS_PER_BATCH \
    --max-sequence-length $MAX_SEQ_LEN \
    --num-kv-cache-slots $NUM_KV_CACHE_SLOTS \
    --warmup 2>&1 | tee "$LOG_FILE"
