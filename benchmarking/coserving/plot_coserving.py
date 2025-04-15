from collections import defaultdict, namedtuple
import os, itertools, json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

def get_tpot(df_original):
    df = df_original.copy()
    # remove entries where is_warmup_request is 1 or decoding_step_idx is < 0
    df = df[(df["is_warmup_request"] == 0) & (df["decoding_step_idx"] >= 0)]
    group = df.groupby("request_guid", as_index=False)
    min_time = group["timestamp"].min()["timestamp"]
    max_time = group["timestamp"].max()["timestamp"]
    num_generated_tokens = group.size()["size"]
    tpots = (max_time - min_time) / num_generated_tokens / 1000
    # return mean and p99 of tpots
    return tpots.mean(), tpots.median(), tpots.quantile(0.99)

def get_throughput(df_original):
    df = df_original.copy()
    # remove entries where is_warmup_request is 1 or request_step_idx is < 0
    df = df[(df["is_warmup_request"] == 0) & (df["decoding_step_idx"] >= 0)]
    # compute the throughput as the number of rows in the filtered dataframe (df) divided by the total time taken
    microsec_to_sec = 1_000_000
    total_time_sec = (df["timestamp"].max() - df["timestamp"].min()) / microsec_to_sec
    total_output_tokens = df.shape[0]
    return total_output_tokens / total_time_sec

def get_ttft(df_original):
    df = df_original.copy()
    # remove entries where is_warmup_request is 1
    df = df[(df["is_warmup_request"] == 0)]
    group = df.groupby("request_guid", as_index=False)
    ttft = group.apply(lambda x: x[x["decoding_step_idx"] == 0]["timestamp"].values[0] - x[x["decoding_step_idx"] == -1]["timestamp"].values[0])/1000
    # convert to milliseconds from microseconds
    return ttft.mean().iloc[1], ttft.median().iloc[1], ttft.quantile(0.99).iloc[1]

def get_queueing_time(df_original):
    df = df_original.copy()
    # remove entries where is_warmup_request is 1
    df = df[(df["is_warmup_request"] == 0)]
    group = df.groupby("request_guid", as_index=False)
    microsec_to_sec = 1_000_000
    # in each group, find the difference between the timestampt at request_step_idx=-1 and the timestamp at request_step_idx=-2.
    queueing_time = group.apply(lambda x: x[x["decoding_step_idx"] == -1]["timestamp"].values[0] - x[x["decoding_step_idx"] == -2]["timestamp"].values[0])/microsec_to_sec
    return queueing_time.mean().iloc[1], queueing_time.median().iloc[1], queueing_time.quantile(0.99).iloc[1]


if __name__ == "__main__":
    # Change directory to that holding this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    models=[m.lower().replace("/", "_") for m in ["meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mistral-Small-24B-Instruct-2501", "meta-llama/Llama-3.1-70B-Instruct"][0:1]]
    tp_degrees=[1,2,4]
    req_prof_data = {}
    step_prof_data = {}

    data_folder=os.path.abspath("../output/coserving/flexllm/profiling")
    

    for i,model in enumerate(models):
        tp_degree=tp_degrees[i]
        fp_req=os.path.join(data_folder, f"inference_request_profiling_sharegpt_{model}_tensor_parallelism_{tp_degree}_max_requests_per_batch_256_max_tokens_per_batch_256_num_kv_cache_slots_90000_qps_0.000000_num_warmup_requests_10.csv")
        req_prof_data[model] = pd.read_csv(fp_req)
        fp_step=os.path.join(data_folder, f"step_profiling_sharegpt_{model}_tensor_parallelism_{tp_degree}_max_requests_per_batch_256_max_tokens_per_batch_256_num_kv_cache_slots_90000_qps_0.000000_num_warmup_requests_10.csv")
        step_prof_data[model] = pd.read_csv(fp_step)

        throughput = get_throughput(req_prof_data[model])
        mean_tpot, median_tpot, tpot99 = get_tpot(req_prof_data[model])
        mean_ttft, median_ttft, ttft99 = get_ttft(req_prof_data[model])
        mean_queueing_time, median_queueing_time, queueing_time99 = get_queueing_time(req_prof_data[model])
        # print(f"Model: {model} - Throughput: {throughput:.3f} tokens/s, Mean TPOT: {mean_tpot:.3f} ms, Mean TTFT: {mean_ttft:.3f} ms, Mean Queueing Time: {mean_queueing_time:.3f} s")

        # Plotting
        # Remove rows where run_idx==0, is_warmup_step==1, or num_decoding_tokens==0
        df = step_prof_data[model].copy()
        df = df[(df["run_idx"] != 0) & (df["is_warmup_step"] != 1) & (df["num_decoding_tokens"] != 0)]
        df = df.sort_values("timestamp").reset_index(drop=True)
        # Calculate time differences in microseconds and convert to seconds
        df["time_diff_sec"] = df["timestamp"].diff() / 1_000_000
        # Remove the first row with NaN time difference
        df = df.dropna()
        # Compute throughput: tokens per second = num_decoding_tokens / time_diff_sec
        df["throughput"] = df["num_decoding_tokens"] / df["time_diff_sec"]
        # Create a relative time axis in seconds (starting from 0)
        df["relative_time"] = (df["timestamp"] - df["timestamp"].iloc[0]) / 1_000_000
        
        # Draw horizontal line at 75% of the throughput and add a text annotation next to it
        throughput_75 = df["throughput"].quantile(0.75)
        # Use a color matching the current model's plot line from the default color cycle
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        plt.axhline(y=throughput_75, color=color, linestyle='--')
        # Position the text near the right end of the plot
        x_pos = df["relative_time"].max() * 0.95
        plt.text(x_pos, throughput_75, f"75%: {throughput_75:.2f} tokens/s", color=color, va='bottom', ha='right')
        print()
        print("Model: ", model)
        print(df["throughput"].describe())

        # print(df["num_finetuning_fwd_tokens"].sum() / df["relative_time"].max())
        # print("Throughput: ", throughput)
        
        plt.plot(df["relative_time"], df["throughput"], label=model)
        plt.grid(True)
        plt.xlabel("Time (s)")
        plt.ylabel("Throughput (tokens/s)")
        plt.title("Throughput vs Time")
        plt.legend()
        # plt.show()
        plt.savefig(f"throughput_vs_time.png")