from collections import defaultdict, namedtuple
import os, itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

IncrDecExpKey = namedtuple('IncrDecExpKey', ['model', 'dataset', 'max_tokens_per_batch', 'max_batch_size'])

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
    return ttft.mean()[1], ttft.median()[1], ttft.quantile(0.99)[1]

def get_queueing_time(df_original):
    df = df_original.copy()
    # remove entries where is_warmup_request is 1
    df = df[(df["is_warmup_request"] == 0)]
    group = df.groupby("request_guid", as_index=False)
    microsec_to_sec = 1_000_000
    # in each group, find the difference between the timestampt at request_step_idx=-1 and the timestamp at request_step_idx=-2.
    queueing_time = group.apply(lambda x: x[x["decoding_step_idx"] == -1]["timestamp"].values[0] - x[x["decoding_step_idx"] == -2]["timestamp"].values[0])/microsec_to_sec
    return queueing_time.mean()[1], queueing_time.median()[1], queueing_time.quantile(0.99)[1]

def plot_throughput(req_prof_data, output_dir="./plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract unique datasets.
    datasets_set = set(key.dataset for key in req_prof_data.keys())

    # Create one plot per dataset and per model.
    for dataset in datasets_set:
        # Find all models for the current dataset.
        models_set = set(key.model for key in req_prof_data.keys() if key.dataset == dataset)
        # Also filter tokens only for the current dataset.
        tokens_set = set(key.max_tokens_per_batch for key in req_prof_data.keys() if key.dataset == dataset)
        for model in models_set:
            plt.figure()
            # For each max_tokens_per_batch, plot a curve over max_batch_size.
            for tokens in sorted(tokens_set):
                batch_sizes = []
                throughputs = []
                # Filter keys for the current dataset, model, and tokens.
                keys = [
                    key for key in req_prof_data.keys() 
                    if key.dataset == dataset and key.model == model and key.max_tokens_per_batch == tokens
                ]
                keys.sort(key=lambda k: k.max_batch_size)
                for key in keys:
                    df = req_prof_data[key]
                    tp = get_throughput(df)
                    batch_sizes.append(key.max_batch_size)
                    throughputs.append(tp)
                if batch_sizes:
                    plt.plot(batch_sizes, throughputs, marker="o",
                             label=f"Max tokens per batch: {tokens}")
            plt.xlabel("Max Batch Size")
            plt.ylabel("Throughput (tokens/sec)")
            plt.title(f"Throughput vs Max Batch Size\ndataset: {dataset}, model: {model}")
            plt.legend()
            plt.ylim(0)
            plt.grid(True)
            plt.tight_layout()
            outfile = os.path.join(output_dir, f"throughput_{dataset}_{model}.png")
            plt.savefig(outfile)
            plt.close()

def plot_tpot(req_prof_data, output_dir="./plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract unique datasets.
    datasets_set = set(key.dataset for key in req_prof_data.keys())

    # Create one plot per dataset and per model.
    for dataset in datasets_set:
        # Find all models for the current dataset.
        models_set = set(key.model for key in req_prof_data.keys() if key.dataset == dataset)
        # Also filter tokens only for the current dataset.
        tokens_set = set(key.max_tokens_per_batch for key in req_prof_data.keys() if key.dataset == dataset)
        for model in models_set:
            plt.figure()
            # For each max_tokens_per_batch, plot a curve over max_batch_size.
            for tokens in sorted(tokens_set):
                batch_sizes = []
                tpots = []
                # Filter keys for the current dataset, model, and tokens.
                keys = [
                    key for key in req_prof_data.keys() 
                    if key.dataset == dataset and key.model == model and key.max_tokens_per_batch == tokens
                ]
                keys.sort(key=lambda k: k.max_batch_size)
                for key in keys:
                    df = req_prof_data[key]
                    tp, _, _ = get_tpot(df)
                    batch_sizes.append(key.max_batch_size)
                    tpots.append(tp)
                if batch_sizes:
                    plt.plot(batch_sizes, tpots, marker="o",
                             label=f"Max tokens per batch: {tokens}")
            plt.xlabel("Max Batch Size")
            plt.ylabel("TPOT (ms)")
            plt.title(f"TPOT vs Max Batch Size\ndataset: {dataset}, model: {model}")
            plt.legend()
            plt.ylim(0)
            plt.grid(True)
            plt.tight_layout()
            outfile = os.path.join(output_dir, f"tpot_{dataset}_{model}.png")
            plt.savefig(outfile)
            plt.close()

def plot_throughput_vs_tpot(req_prof_data, output_dir="./plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract unique datasets.
    datasets_set = set(key.dataset for key in req_prof_data.keys())
    
    for dataset in datasets_set:
        # Get models for the current dataset.
        models_set = set(key.model for key in req_prof_data.keys() if key.dataset == dataset)
        for model in models_set:
            plt.figure()
            # Filter keys for the current dataset and model.
            keys_dm = [key for key in req_prof_data.keys() 
                       if key.dataset == dataset and key.model == model]
            
            # Group keys by max_batch_size.
            batch_size_groups = {}
            for key in keys_dm:
                batch_size_groups.setdefault(key.max_batch_size, []).append(key)
            
            # For every max_batch_size group, sort by max_tokens_per_batch and compute points.
            for batch_size, keys_group in batch_size_groups.items():
                keys_group.sort(key=lambda k: k.max_tokens_per_batch)
                tpot_values = []
                throughput_values = []
                token_values = []
                for key in keys_group:
                    df = req_prof_data[key]
                    # Use the first element of the get_tpot output (TPOT mean).
                    tp, _, _ = get_tpot(df)
                    thp = get_throughput(df)
                    tpot_values.append(tp)
                    throughput_values.append(thp)
                    token_values.append(key.max_tokens_per_batch)
                
                # Plot the line for this max_batch_size.
                plt.plot(tpot_values, throughput_values, marker="o", linestyle="-",
                         label=f"Max Batch Size: {batch_size}")
                
                # Annotate each point with the max_tokens_per_batch value.
                for x, y, tokens in zip(tpot_values, throughput_values, token_values):
                    plt.annotate(f"{tokens}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
            
            plt.xlabel("TPOT (ms)")
            plt.ylabel("Throughput (tokens/sec)")
            plt.title(f"Throughput vs TPOT\nDataset: {dataset}, Model: {model}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            outfile = os.path.join(output_dir, f"throughput_vs_tpot_{dataset}_{model}.png")
            plt.savefig(outfile)
            plt.close()


if __name__ == "__main__":
    models=["meta-llama/Llama-3.1-8B-Instruct".lower().replace("/", "_")]
    datasets=["sharegpt", "wildchat"]
    # datasets=["sharegpt"]
    max_tokens_per_batch=[128, 256, 512, 1024, 2048]
    max_batch_size=[4, 8, 16, 32, 64, 128, 256]

    req_prof_data = {}
    step_prof_data = {}

    # Change directory to that holding this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Iterate over every combination of the possible key values.
    for model, dataset, tokens, batch in itertools.product(models, datasets, max_tokens_per_batch, max_batch_size):
        folder = "/pscratch/sd/g/goliaro/output/incr_decoding/8B/profiling/"
        req_prefix = "inference_request_profiling_"
        step_prefix = "step_profiling_"
        fp = f"{dataset}_{model}_tensor_parallelism_1_max_requests_per_batch_{batch}_max_tokens_per_batch_{tokens}_num_kv_cache_slots_400000_qps_0.000000_num_warmup_requests_10.csv"
        key = IncrDecExpKey(model, dataset, tokens, batch)
        req_prof_data[key] = pd.read_csv(os.path.join(folder, req_prefix + fp))
        step_prof_data[key] = pd.read_csv(os.path.join(folder, step_prefix + fp))

    plot_throughput(req_prof_data)
    plot_tpot(req_prof_data)
    plot_throughput_vs_tpot(req_prof_data)