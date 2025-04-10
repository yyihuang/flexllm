#!/bin/bash
#SBATCH --account=m4138
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --constraint gpu,ss11,a100,hbm80g
#SBATCH --time=00:30:00
#SBATCH --job-name=vllm_benchmark
#SBATCH --output=/global/homes/g/goliaro/flexllm/benchmarking/output/slurm/%x_%A_%a.out
#SBATCH --error=/global/homes/g/goliaro/flexllm/benchmarking/output/slurm/%x_%A_%a.err
#SBATCH --array=0-23

set -x
set -o pipefail

# Change directory to the script’s location
CWD_="/global/homes/g/goliaro/flexllm/benchmarking/incr_dec"
cd "${CWD_}"
export PYTHONPATH="$(realpath $PWD/../../vllm)"

# Define arrays (the order of eager_mode is the same as in the original loop)
VLLM_V1=(0 1)
MODEL_NAMES=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-32B-Instruct"
  "meta-llama/Llama-3.1-70B-Instruct"
)
TP_DEGREES=(1 2 4)
TRACES=(sharegpt wildchat)
EAGER_MODE=(true false)

# Function to check for available GPUs.
check_gpus() {
  declare -g gpu_count
  gpu_count=$(nvidia-smi --list-gpus | wc -l)
  if [[ $gpu_count -gt 0 ]]; then
    echo "GPU found."
  else
    echo "Need at least 1 GPU to run benchmarking."
    exit 1
  fi
  declare -g gpu_type
  gpu_type=$(nvidia-smi --query-gpu=name --format=csv,noheader | awk '{print $2}')
  echo "GPU type is $gpu_type"
}

# Wait until the vllm server is responsive.
wait_for_server() {
  timeout 1200 bash -c '
    until curl -X POST localhost:8000/v1/completions; do
      sleep 10
    done' && return 0 || return 1
}

# Kill processes using the GPU and clean up.
kill_gpu_processes() {
  lsof -t -i:8000 | xargs -r kill -9
  pgrep python3 | xargs -r kill -9

  # wait until GPU memory usage is below 1GB
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)" -ge 1000 ]; do
    sleep 1
  done

  # remove vllm config file
  rm -rf ~/.config/vllm
}

# Run the server and client for a given parameter combination.
run_serving_tests() {
  local model_name=$1
  local tp_degree=$2
  local trace=$3
  local vllm_use_v1=$4
  local eager_mode=$5

  server_command="VLLM_USE_V1=${vllm_use_v1} python3 \
      -m vllm.entrypoints.openai.api_server \
      --model ${model_name} \
      --tensor-parallel-size ${tp_degree} \
      --swap-space 0 \
      --disable-log-stats \
      --disable-log-requests"
  
  if [ "$eager_mode" = true ]; then
    server_command+=" --enforce-eager"
  fi
  echo "Starting VLLM server"
  echo "Server command: $server_command"
  bash -c "$server_command" &
  server_pid=$!

  # Wait until the server is up.
  if wait_for_server; then
    echo ""
    echo "vllm server is up and running."
  else
    echo ""
    echo "vllm failed to start within the timeout period."
  fi

  mkdir -p ../output/vllm

  # Construct the result filename and convert it to lowercase.
  result_filename=$(echo "results_${trace}_$( [ "$eager_mode" = true ] && echo "eager_" )$( [ "$vllm_use_v1" = 1 ] && echo "v1_" )${model_name//\//_}.json" | tr '[:upper:]' '[:lower:]')

  # Build the client command with the result_filename variable.
  client_command="VLLM_USE_V1=${vllm_use_v1} PYTHONPATH=${PYTHONPATH} python3 benchmark_vllm.py \
        --model ${model_name} \
        --backend vllm \
        --ignore-eos \
        --dataset-path ../traces/${trace}.json \
        --save-result \
        --result-dir ../output/vllm \
        --result-filename ${result_filename}"

  echo "Client command: $client_command"
  bash -c "$client_command"

  # Clean up.
  kill -9 $server_pid
  kill_gpu_processes
}

# Main: run prerequisite checks then compute indices from SLURM_ARRAY_TASK_ID.
main() {
  check_gpus
  (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
  (which jq) || (apt-get update && apt-get -y install jq)
  (which lsof) || (apt-get update && apt-get install -y lsof)

  # Set up environment variables required by the client.
  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
  export VLLM_LOG_LEVEL="WARNING"

  # Total combinations: 2 (TRACES) * 2 (VLLM_V1) * 2 (EAGER_MODE) * 3 (MODEL_NAMES) = 24.
  # Decompose SLURM_ARRAY_TASK_ID (0-23) into four indices.
  job_id=${SLURM_ARRAY_TASK_ID}
  # Each trace covers 12 tasks.
  trace_idx=$(( job_id / (2 * 2 * 3) ))      # job_id / 12
  remainder=$(( job_id % (2 * 2 * 3) ))
  # Each VLLM_V1 covers 6 tasks.
  vllm_idx=$(( remainder / (2 * 3) ))          # remainder / 6
  remainder2=$(( remainder % (2 * 3) ))
  # Each eager_mode covers 3 tasks.
  eager_idx=$(( remainder2 / 3 ))
  model_idx=$(( remainder2 % 3 ))

  trace=${TRACES[$trace_idx]}
  vllm_use_v1=${VLLM_V1[$vllm_idx]}
  eager_mode_val=${EAGER_MODE[$eager_idx]}
  model_name="${MODEL_NAMES[$model_idx]}"
  tp_degree="${TP_DEGREES[$model_idx]}"

  echo "Parameters for task $job_id:"
  echo "  Trace: $trace"
  echo "  VLLM_USE_V1: $vllm_use_v1"
  echo "  Eager Mode: $eager_mode_val"
  echo "  Model: $model_name"
  echo "  TP Degree: $tp_degree"

  run_serving_tests "$model_name" "$tp_degree" "$trace" "$vllm_use_v1" "$eager_mode_val"
}

main
