from collections import defaultdict, namedtuple
import os, itertools, json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

def get_throughput(df_original):
    df = df_original.copy()
    # remove entries where is_warmup_request is 1 or run_idx is <= 0
    df = df[(df["is_warmup_step"] == 0) & (df["run_idx"] > 0)]
    # remove entries where both num_finetuning_fwd_tokens and num_finetuning_bwd_tokens are 0
    df = df[(df["num_finetuning_fwd_tokens"] > 0) | (df["num_finetuning_bwd_tokens"] > 0)]
    # compute the throughput as the number of rows in the filtered dataframe (df) divided by the total time taken
    microsec_to_sec = 1_000_000
    total_time_sec = (df["timestamp"].max() - df["timestamp"].min()) / microsec_to_sec
    # check that the sum of num_finetuning_fwd_tokens and num_finetuning_bwd_tokens is equal
    assert df["num_finetuning_fwd_tokens"].sum() == df["num_finetuning_bwd_tokens"].sum(), "num_finetuning_fwd_tokens and num_finetuning_bwd_tokens do not match"
    # set total_processed_tokens to be the sum of num_finetuning_fwd_tokens
    total_processed_tokens = df["num_finetuning_fwd_tokens"].sum()
    print("Total processed tokens: ", total_processed_tokens)
    return total_processed_tokens / total_time_sec

step_profiling = pd.read_csv("/global/homes/g/goliaro/flexllm/benchmarking/output/finetuning/8B/profiling/step_profiling_unknown_meta-llama_llama-3.1-8b-instruct_tensor_parallelism_1_max_requests_per_batch_1_max_tokens_per_batch_8192_num_kv_cache_slots_8192_qps_0.000000_num_warmup_requests_10.csv")

throughput = get_throughput(step_profiling)
print(f"Throughput: {throughput} tokens/sec")
