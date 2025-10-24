import argparse
import json
import sys
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.table import Table

from metric import A100_80_GPU_COST, A100_GPU_COST


def main(data_folder_path: Path):
    # Find all client files and the server file
    file = data_folder_path / "server_only.jsonl"

    if not file.exists():
        print(f"Error: Server file not found: {file}")
        sys.exit(1)

    with open(file, "r") as f:
        try:
            raw_data = [json.loads(line) for line in f.readlines()]
        except json.JSONDecodeError as e:
            print(f"Error: Error decoding JSON from {file}: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: Error reading file {file}: {e}")
            sys.exit(1)

    # Rest of the processing remains the same, operating on the combined raw_data
    df = pl.json_normalize(raw_data).drop("timestamp", strict=False)

    return df


def overall_analysis(df: pl.DataFrame):
    return {
        "draft": {
            "end_to_end": {
                "non-prefill": (
                    df.filter(pl.col("iter_idx") != 0)
                    .select("draft.end_to_end")
                    .mean()
                    .item(),
                    df.filter(pl.col("iter_idx") != 0)
                    .select("draft.end_to_end")
                    .std()
                    .item(),
                ),
                "prefill": (
                    df.filter(pl.col("iter_idx") == 0)
                    .select("draft.end_to_end")
                    .mean()
                    .item(),
                    df.filter(pl.col("iter_idx") == 0)
                    .select("draft.end_to_end")
                    .std()
                    .item(),
                ),
            }
        },
        "target": {
            "end_to_end": {
                "non-prefill": (
                    df.filter(pl.col("prefill") == 0)
                    .select("target.end_to_end")
                    .mean()
                    .item(),
                    df.filter(pl.col("prefill") == 0)
                    .select("target.end_to_end")
                    .std()
                    .item(),
                ),
                "prefill": (
                    df.filter(pl.col("prefill") != 0)
                    .select("target.end_to_end")
                    .mean()
                    .item(),
                    df.filter(pl.col("prefill") != 0)
                    .select("target.end_to_end")
                    .std()
                    .item(),
                ),
            }
        },
        "overall": {
            "non-prefill": (
                df.filter(pl.col("prefill") == 0)
                .select(pl.col("draft.end_to_end") + pl.col("target.end_to_end"))
                .mean()
                .item(),
                df.filter(pl.col("prefill") == 0)
                .select(pl.col("draft.end_to_end") + pl.col("target.end_to_end"))
                .std()
                .item(),
            ),
            "prefill": (
                df.filter(pl.col("iter_idx") == 0)
                .select(pl.col("draft.end_to_end") + pl.col("target.end_to_end"))
                .mean()
                .item(),
                df.filter(pl.col("iter_idx") == 0)
                .select(pl.col("draft.end_to_end") + pl.col("target.end_to_end"))
                .std()
                .item(),
            ),
        },
        "tokens": {
            "generated": df.filter(pl.col("prefill") == 0)
            .select("num_accepted_tokens")
            .sum()
            .item(),
            "accepted": (
                df.select("num_accepted_tokens").mean().item(),
                df.select("num_accepted_tokens").std().item(),
            ),
        },
        "latency": {
            "value": df.filter((pl.col("prefill") == 0))
            .select(pl.col("draft.end_to_end") + pl.col("target.end_to_end"))
            .sum()
            .item()
            / df.filter((pl.col("prefill") == 0))
            .select(pl.col("num_accepted_tokens"))
            .sum()
            .item(),
        },
        "running_time": {
            "server": (
                df.group_by("server_iter_idx")
                .agg(
                    pl.first("draft.end_to_end").alias("draft_end_to_end"),
                    pl.first("target.end_to_end").alias("target_end_to_end"),
                )
                .select(pl.col("draft_end_to_end") + pl.col("target_end_to_end"))
            )
            .sum()
            .item()
        },
        "throughput": (
            df.filter(pl.col("prefill") == 0).select("num_accepted_tokens").sum().item()
            / (
                df.filter(pl.col("prefill") == 0)
                .group_by("server_iter_idx")
                .agg(
                    pl.first("draft.end_to_end").alias("draft_end_to_end"),
                    pl.first("target.end_to_end").alias("target_end_to_end"),
                )
                .select(pl.col("draft_end_to_end") + pl.col("target_end_to_end"))
                .sum()
                .item()
                / 1_000  # convert to seconds
            )
        ),
        "cost": {
            "server": (
                df.filter(pl.col("prefill") == 0)
                .group_by("server_iter_idx")
                .agg(
                    pl.first("draft.end_to_end").alias("draft_end_to_end"),
                    pl.first("target.end_to_end").alias("target_end_to_end"),
                )
                .select(pl.col("draft_end_to_end") + pl.col("target_end_to_end"))
            )
            .sum()
            .item()
            * GPU_COST
            / 1000,
        },
        "cost_efficiency": (
            df.filter(pl.col("prefill") == 0).select("num_accepted_tokens").sum().item()
            / (
                df.filter(pl.col("prefill") == 0)
                .group_by("server_iter_idx")
                .agg(
                    pl.first("draft.end_to_end").alias("draft_end_to_end"),
                    pl.first("target.end_to_end").alias("target_end_to_end"),
                )
                .select(pl.col("draft_end_to_end") + pl.col("target_end_to_end"))
                .sum()
                .item()
                / 1_000  # convert to seconds
                * GPU_COST
            )
            / 1_000  # convert to 1k tokens
        ),
    }


def pprint(df: pl.DataFrame):
    console = Console()

    overall_table = Table(title="Overall")

    overall_table.add_column("Metric", justify="left")
    overall_table.add_column("Value", justify="right", min_width=20)
    overall_table.add_column("Std", justify="right", min_width=20)

    overall_metrics = overall_analysis(df)

    overall_table.add_row(
        "Draft (prefill)",
        f"{overall_metrics['draft']['end_to_end']['prefill'][0]:.3f} ms",
        f"{overall_metrics['draft']['end_to_end']['prefill'][1]:.3f} ms",
    )
    overall_table.add_row(
        "Draft (non-prefill)",
        f"{overall_metrics['draft']['end_to_end']['non-prefill'][0]:.3f} ms",
        f"{overall_metrics['draft']['end_to_end']['non-prefill'][1]:.3f} ms",
    )
    overall_table.add_section()
    overall_table.add_row(
        "Target (prefill)",
        f"{overall_metrics['target']['end_to_end']['prefill'][0]:.3f} ms",
        f"{overall_metrics['target']['end_to_end']['prefill'][1]:.3f} ms",
    )
    overall_table.add_row(
        "Target (non-prefill)",
        f"{overall_metrics['target']['end_to_end']['non-prefill'][0]:.3f} ms",
        f"{overall_metrics['target']['end_to_end']['non-prefill'][1]:.3f} ms",
    )

    overall_table.add_section()
    overall_table.add_row(
        "Overall (prefill)",
        f"{overall_metrics['overall']['prefill'][0]:.3f} ms",
        f"{overall_metrics['overall']['prefill'][1]:.3f} ms",
    )
    overall_table.add_row(
        "Overall (non-prefill)",
        f"{overall_metrics['overall']['non-prefill'][0]:.3f} ms",
        f"{overall_metrics['overall']['non-prefill'][1]:.3f} ms",
    )
    overall_table.add_section()
    overall_table.add_row(
        "Accept Tokens",
        f"{overall_metrics['tokens']['accepted'][0]:.2f}",
        f"{overall_metrics['tokens']['accepted'][1]:.2f}",
    )
    overall_table.add_section()
    overall_table.add_row(
        "Inter token latency",
        f"{overall_metrics['latency']['value']:.3f} ms/tok",
        "",
    )
    overall_table.add_section()
    overall_table.add_row(
        "Server Running Time",
        f"{overall_metrics['running_time']['server'] / 1000:.3f} s",
        "",
    )
    overall_table.add_row(
        "Server cost",
        f"${overall_metrics['cost']['server']:.3f}",
        "",
    )
    overall_table.add_row(
        "Generated tokens",
        f"{overall_metrics['tokens']['generated']}",
        "",
    )
    # Calculate cost per 1M tokens, handling division by zero
    total_cost = overall_metrics['cost']['server']
    total_tokens = overall_metrics['tokens']['generated']
    cost_per_1m_tokens = total_cost / total_tokens * 1_000_000

    overall_table.add_row(
        "Dollars per 1M tokens",
        f"${cost_per_1m_tokens:.3f}",
        "",
    )
    overall_table.add_row(
        "Throughput",
        f"{overall_metrics['throughput']:.3f} tokens/s",
        "",
    )
    overall_table.add_row(
        "Cost efficiency",
        f"{overall_metrics['cost_efficiency']:.3f} 1k tokens/s",
        "",
    )

    console.print(overall_table)


def plain_text_print(df: pl.DataFrame):
    """
    Print the metrics in a plain text format.
    Some metrics are not printed in plain text format due to simplification.
    """
    metrics = overall_analysis(df)

    values = [
        # Client Draft Latency (ms)
        f"{metrics['draft']['end_to_end']['prefill'][0]:.3f}",  # prefill_mean
        f"{metrics['draft']['end_to_end']['prefill'][1]:.3f}",  # prefill_std
        f"{metrics['draft']['end_to_end']['non-prefill'][0]:.3f}",  # non-prefill_mean
        f"{metrics['draft']['end_to_end']['non-prefill'][1]:.3f}",  # non-prefill_std
        "",  # proactive_mean
        "",  # proactive_std
        # Client Target Latency (ms)
        f"{metrics['target']['end_to_end']['prefill'][0]:.3f}",  # prefill_mean
        f"{metrics['target']['end_to_end']['prefill'][1]:.3f}",  # prefill_std
        f"{metrics['target']['end_to_end']['non-prefill'][0]:.3f}",  # non-prefill_mean
        f"{metrics['target']['end_to_end']['non-prefill'][1]:.3f}",  # non-prefill_std
        "",  # proactive_mean
        "",  # proactive_std
        # Server Target Latency (ms)
        "",  # prefill_mean
        "",  # prefill_std
        "",  # non-prefill_mean
        "",  # non-prefill_std
        # Client Overall Latency (ms)
        f"{metrics['overall']['prefill'][0]:.3f}",  # prefill_mean
        f"{metrics['overall']['prefill'][1]:.3f}",  # prefill_std
        f"{metrics['overall']['non-prefill'][0]:.3f}",  # non-prefill_mean
        f"{metrics['overall']['non-prefill'][1]:.3f}",  # non-prefill_std
        "",  # proactive_mean
        "",  # proactive_std
        # Proactive Ratio (%)
        "",  # proactive ratio
        # Accepted Tokens per step (tokens)
        f"{metrics['tokens']['accepted'][0]:.2f}",
        f"{metrics['tokens']['accepted'][1]:.2f}",
        # Client Inter-token Latency (non-prefill) (ms/tok)
        f"{metrics['latency']['value']:.3f}",
        # Server Total Running Time (s)
        f"{metrics['running_time']['server'] / 1000:.3f}",
        # Server Total Cost (Numeric Value)
        f"{metrics['cost']['server']:.3f}",
        # Client Total Processing Time (s)
        "",  # client processing time
        # Client Total Cost (Numeric Value)
        "",  # client cost
        # Total Accepted Tokens (tokens)
        f"{metrics['tokens']['generated']}",
    ]

    # Calculate Overall Cost per 1M Accepted Tokens and append
    total_cost_val = metrics["cost"]["server"]
    total_generated_tokens_val = metrics["tokens"]["generated"]
    cost_per_1m_tokens_val = (
        (total_cost_val / total_generated_tokens_val * 1000000)
        if total_generated_tokens_val > 0
        else 0.0
    )
    values.append(f"{cost_per_1m_tokens_val:.3f}")

    print("\t".join(values))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Path to the data file")
    parser.add_argument("--plain", action="store_true", help="Use plain text data")
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
        raise ValueError(f"Data path '{data_folder_path}' is not a valid directory")

    df = main(data_folder_path)

    if args.plain:
        plain_text_print(df)
    else:
        pprint(df)
