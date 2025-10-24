import argparse
import json
import sys
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.table import Table

from metric import A100_80_GPU_COST, A100_GPU_COST, RTX4090_GPU_COST  # noqa: F401


def load_data(data_folder_path: Path):
    file = data_folder_path / "auto.jsonl"

    with open(file, "r") as f:
        try:
            raw_data = [json.loads(line) for line in f.readlines()]
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            sys.exit(1)

    if not raw_data:
        print("No data found in the file.")
        sys.exit(1)

    df = pl.json_normalize(raw_data).drop("timestamp", strict=False)
    return df


def overall_analysis(df: pl.DataFrame):
    return {
        "forward": {
            "prefill": (
                df.filter(pl.col("prefill_cnt") != 0).select("forward_t").mean().item(),
                df.filter(pl.col("prefill_cnt") != 0).select("forward_t").std().item(),
            ),
            "non-prefill": (
                df.filter(pl.col("prefill_cnt") == 0).select("forward_t").mean().item(),
                df.filter(pl.col("prefill_cnt") == 0).select("forward_t").std().item(),
            ),
        },
        "running_time": df.filter(pl.col("prefill_cnt") == 0)
        .group_by("server_step_idx")
        .agg(pl.first("forward_t"))
        .select("forward_t")
        .sum()
        .item(),
        "cost": df.filter(pl.col("prefill_cnt") == 0)
        .group_by("server_step_idx")
        .agg(pl.first("forward_t"))
        .select("forward_t")
        .sum()
        .item()
        * GPU_COST
        / 1000,  # Convert to seconds
        "tokens": df.filter(pl.col("prefill_cnt") == 0)
        .select("forward_t")
        .count()
        .item(),
    }


def main(data: Path):
    data_path = data / "auto.jsonl"

    with open(data_path, "r") as f:
        raw_data = [json.loads(line) for line in f.readlines()]

    df = pl.json_normalize(raw_data).drop("timestamp", strict=False)

    forward_t = round(df.select("forward_t").mean().item(), 3)
    forward_t_std = round(df.select("forward_t").std().item(), 3)

    print(f"Mean forward time: {forward_t}ms, std: {forward_t_std}ms")


def pprint(df: pl.DataFrame):
    console = Console()
    overall_table = Table(title="Overall")

    overall_table.add_column("Metric", justify="left")
    overall_table.add_column("Value", justify="right", min_width=20)
    overall_table.add_column("Std", justify="right", min_width=20)

    overall_metrics = overall_analysis(df)

    overall_table.add_row(
        "Forward (prefill)",
        f"{overall_metrics['forward']['prefill'][0]:.3f}ms",
        f"{overall_metrics['forward']['prefill'][1]:.3f}ms",
    )
    overall_table.add_row(
        "Forward (non-prefill)",
        f"{overall_metrics['forward']['non-prefill'][0]:.3f}ms",
        f"{overall_metrics['forward']['non-prefill'][1]:.3f}ms",
    )
    overall_table.add_section()
    overall_table.add_row(
        "Running time",
        f"{overall_metrics['running_time'] / 1000:.3f}s",
    )
    overall_table.add_row(
        "Cost",
        f"${overall_metrics['cost'] / overall_metrics['tokens'] * 1000000:.3f}",
    )

    console.print(overall_table)


def plain_text_print(df: pl.DataFrame):
    overall_metrics = overall_analysis(df)

    values = [
        # Client Draft Latency (ms):
        "",  # prefill_mean
        "",  # prefill_std
        "",  # non-prefill_mean
        "",  # non-prefill_std,
        "",  # proactive_mean
        "",  # proactive_std
        # Client Target Latency (ms)
        "",  # prefill_mean
        "",  # prefill_std
        "",  # non-prefill_mean
        "",  # non-prefill_std
        "",  # proactive_mean
        "",  # proactive_std
        # Server Target Latency (ms)
        f"{overall_metrics['forward']['prefill'][0]:.3f}",  # prefill_mean
        f"{overall_metrics['forward']['prefill'][1]:.3f}",  # prefill_std
        f"{overall_metrics['forward']['non-prefill'][0]:.3f}",  # non-prefill_mean
        f"{overall_metrics['forward']['non-prefill'][1]:.3f}",  # non-prefill_std
        # Client Overall Latency (ms)
        f"{overall_metrics['forward']['prefill'][0]:.3f}",  # prefill_mean
        f"{overall_metrics['forward']['prefill'][1]:.3f}",  # prefill_std
        f"{overall_metrics['forward']['non-prefill'][0]:.3f} ",  # non-prefill_mean
        f"{overall_metrics['forward']['non-prefill'][1]:.3f}",  # non-prefill_std
        "",  # proactive_mean
        "",  # proactive_std
        # Proactive Ratio (%)
        "",
        # Accepted Tokens per step (tokens)
        "1.00",  # mean
        "0.00",  # std
        # Client Inter-token Latency (non-prefill) (ms/tok)
        f"{overall_metrics['forward']['non-prefill'][0]:.3f}",
        # Server Total Running Time (s)
        f"{overall_metrics['running_time'] / 1000:.3f}",
        # Server Total Cost (Numeric Value)
        f"{overall_metrics['cost']:.3f}",
        # Client Total Processing Time (s)
        "",
        # Client Total Cost (Numeric Value)
        "",
        # Total Accepted Tokens (tokens)
        "",
    ]

    # Calculate Overall Cost per 1M Accepted Tokens and append
    cost_per_1m_tokens_val = (
        overall_metrics["cost"] / overall_metrics["tokens"] * 1_000_000
    )
    values.append(f"{cost_per_1m_tokens_val:.3f}")

    print("\t".join(values))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Path to the data file")
    parser.add_argument("--plain", action="store_true", help="Use plain text output")
    parser.add_argument("--gpu", default="A100_80", type=str, choices=["A100_80", "A100_40"])
    args = parser.parse_args()
    
    if args.gpu == "A100_80":
        print("Using A100_80 GPU")
        GPU_COST = A100_80_GPU_COST
    elif args.gpu == "A100_40":
        print("Using A100_40 GPU")
        GPU_COST = A100_GPU_COST
    else:
        raise ValueError("Invalid GPU option")

    data_folder_path = Path(args.data)

    if not data_folder_path.is_dir():
        raise ValueError("Data path is not a valid directory")

    df = load_data(data_folder_path)

    if args.plain:
        plain_text_print(df)
    else:
        pprint(df)
