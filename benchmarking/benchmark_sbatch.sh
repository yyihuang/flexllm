#!/bin/bash
set -ex

# Compute the build directory (assumes this script is in e.g. scripts/ folder)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="$( realpath "${SCRIPT_DIR}/../flexflow-serve/build" )"
cd "$BUILD_DIR"

# Activate the python environment
source ./set_python_envs.sh

# Define experiment parameters
MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"
NGPUS=4
NCPUS=16
FSIZE=76000
ZSIZE=200000

OUTPUT_FOLDER="../../benchmarking/output/incr_decoding"
TRACES_FOLDER="../../benchmarking/traces"
MAX_SEQ_LEN=8192
NUM_KV_CACHE_SLOTS=$(( MAX_SEQ_LEN * 4 ))

batch_sizes=(128 64 32 16 8 4)
trace_files=(sharegpt wildchat)
max_tokens_per_batch_values=(2048 1024 512 256 128)

mkdir -p "$OUTPUT_FOLDER/out" "$OUTPUT_FOLDER/logs" "$OUTPUT_FOLDER/profiling"

# Loop over experiments and submit each as an independent sbatch job.
for i in "${!trace_files[@]}"; do
  for k in "${!batch_sizes[@]}"; do
    for j in "${!max_tokens_per_batch_values[@]}"; do
      
      TRACE_FILE="${TRACES_FOLDER}/${trace_files[$i]}.json"
      if [ ! -f "$TRACE_FILE" ]; then
          echo "File $TRACE_FILE not found"
          exit 1
      fi
      
      MAX_TOKENS_PER_BATCH=${max_tokens_per_batch_values[$j]}
      BATCH_SIZE=${batch_sizes[$k]}
      
      # Create a job name (make sure it fits your naming conventions)
      JOB_NAME="incr_dec_${trace_files[$i]}_bz_${BATCH_SIZE}_tokens_${MAX_TOKENS_PER_BATCH}"
      
      # Set output file names
      OUTPUT_FILE="${OUTPUT_FOLDER}/output/${JOB_NAME}_kv_cache_slots_${NUM_KV_CACHE_SLOTS}.json"
      LOG_FILE="${OUTPUT_FOLDER}/logs/${JOB_NAME}_kv_cache_slots_${NUM_KV_CACHE_SLOTS}.out"
      
      # Submit the job via sbatch using a heredoc.
      sbatch <<EOF
#!/bin/bash
#SBATCH --account=m4138
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --constraint gpu,ss11,a100,hbm80g
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_FILE}
#SBATCH --error=${LOG_FILE}
#SBATCH --time=02:00:00

# Change to the build directory (passed from the master script)
cd "${BUILD_DIR}"
source ./set_python_envs.sh

echo "Running ${TRACE_FILE} with BATCH_SIZE=${BATCH_SIZE} and MAX_TOKENS_PER_BATCH=${MAX_TOKENS_PER_BATCH} and NUM_KV_CACHE_SLOTS=${NUM_KV_CACHE_SLOTS}"

# Run your experiment
time ./inference/incr_decoding/incr_decoding \\
    -ll:cpu ${NCPUS} -ll:gpu ${NGPUS} -ll:util ${NCPUS} \\
    -ll:fsize ${FSIZE} -ll:zsize ${ZSIZE} \\
    -llm-model ${MODEL_NAME} --fusion \\
    -tensor-parallelism-degree ${NGPUS} \\
    -prompt ${TRACE_FILE} \\
    -output-file ${OUTPUT_FILE} \\
    -profiling-folder "${OUTPUT_FOLDER}/profiling" \\
    --max-requests-per-batch ${BATCH_SIZE} \\
    --max-tokens-per-batch ${MAX_TOKENS_PER_BATCH} \\
    --max-sequence-length ${MAX_SEQ_LEN} \\
    --num-kv-cache-slots ${NUM_KV_CACHE_SLOTS} \\
    --warmup 2>&1 | tee ${LOG_FILE}
EOF

    done
  done
done