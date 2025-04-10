#!/bin/bash

set -x
set -o pipefail

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"
export PYTHONPATH="$(realpath $PWD/../../vllm)"

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
EAGER_MODES=(
  true
  false
)
BATCH_SIZE=256
MAX_TOKENS_PER_BATCH=256
VLLM_USE_V1=1

check_gpus() {
  # check the number of GPUs and GPU type.
  declare -g gpu_count=$(nvidia-smi --list-gpus | wc -l)
  if [[ $gpu_count -gt 0 ]]; then
    echo "GPU found."
  else
    echo "Need at least 1 GPU to run benchmarking."
    exit 1
  fi
  declare -g gpu_type=$(nvidia-smi --query-gpu=name --format=csv,noheader | awk '{print $2}')
  echo "GPU type is $gpu_type"
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  timeout 1200 bash -c '
    until curl -X POST localhost:8000/v1/completions; do
      sleep 10
    done' && return 0 || return 1
}

kill_gpu_processes() {
  lsof -t -i:8000 | xargs -r kill -9
  pgrep python3 | xargs -r kill -9


  # wait until GPU memory usage smaller than 1GB
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)" -ge 1000 ]; do
    sleep 1
  done

  # remove vllm config file
  rm -rf ~/.config/vllm

}

run_serving_tests() {
  local model_name=$1
  local tp_degree=$2
  local trace=$3
  local vllm_use_v1=$4
  local eager_mode=$5
  local batch_size=$6
  local max_num_batched_tokens=$7

  # if the max_num_batched_tokens is less than the batch_size, return
  if [ "$max_num_batched_tokens" -lt "$batch_size" ]; then
    echo "max_num_batched_tokens is less than batch_size, skipping this test."
    return
  fi

  server_command="VLLM_USE_V1=${vllm_use_v1} python3 \
      -m vllm.entrypoints.openai.api_server \
      --model ${model_name} \
      --tensor-parallel-size ${tp_degree} \
      --enable-chunked-prefill \
      --max-num-seqs ${batch_size} \
      --max-num-batched-tokens ${max_num_batched_tokens} \
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

  # wait until the server is alive
  if wait_for_server; then
    echo ""
    echo "vllm server is up and running."
  else
    echo ""
    echo "vllm failed to start within the timeout period."
  fi

  mkdir -p ../output/vllm

  # Construct the result filename and convert it to lowercase.
  result_filename=$(echo "results_${trace}_$( [ "$eager_mode" = true ] && echo "eager_" )$( [ "$vllm_use_v1" = 1 ] && echo "v1_" )${model_name//\//_}_bz_${batch_size}_max_num_batched_tokens_${max_num_batched_tokens}.json" | tr '[:upper:]' '[:lower:]')

  # Build the client command with the result_filename variable.
  client_command="VLLM_USE_V1=${vllm_use_v1} PYTHONPATH=${PYTHONPATH} python3 benchmark_vllm.py \
        --model ${model_name} \
        --backend vllm \
        --ignore-eos \
        --dataset-path ../traces/${trace}.json \
        --save-result --save-detailed \
        --result-dir ../output/vllm \
        --result-filename ${result_filename}"

  echo "Client command: $client_command"
  bash -c "$client_command"

  # clean up
  kill -9 $server_pid
  kill_gpu_processes
}

main() {
    check_gpus
    (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
    (which jq) || (apt-get update && apt-get -y install jq)
    (which lsof) || (apt-get update && apt-get install -y lsof)

    # get the current IP address, required by benchmark_serving.py
    export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
    # turn of the reporting of the status of each request, to clean up the terminal output
    export VLLM_LOG_LEVEL="WARNING"

    for trace in "${TRACES[@]}"; do
    for eager_mode in "${EAGER_MODES[@]}"; do
    for i in "${!MODEL_NAMES[@]}"; do
      model_name="${MODEL_NAMES[$i]}"
      tp_degree="${TP_DEGREES[$i]}"
      run_serving_tests "$model_name" "$tp_degree" "$trace" "$VLLM_USE_V1" "$eager_mode" "$BATCH_SIZE" "$MAX_TOKENS_PER_BATCH"
    done
    done
    done

}

main
