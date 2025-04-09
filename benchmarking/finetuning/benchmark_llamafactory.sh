set -x
set -e

# Cd into the LLaMA-Factory main directory
cd "${BASH_SOURCE[0]%/*}/../../LLaMA-Factory"

# rm -rf saves

# Single GPU LLAMA-3.1 8B
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/flexllm/t1_8B.yaml
# Multi GPU (2) QWEN-2.5 32B
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train examples/flexllm/t1_32B.yaml
# Multi GPU (4) LLAMA-3.1 70B
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/flexllm/t1_70B.yaml
