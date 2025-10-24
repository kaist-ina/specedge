import argparse
import os
import subprocess
from pathlib import Path
from typing import Optional

import torch
import yaml
from rich.progress import Progress

import log
import util
from config import AutoregressiveBatchConfig as config
from specedge.engine.graph import BatchGraphEngine


def main():
    logger = log.get_logger()
    result_logger = log.get_result_logger()

    logger.info("Starting %s", config.exp_name)
    logger.info(
        "Initializing %s on %s with dtype %s", config.model, config.device, config.dtype
    )

    model = util.load_graph_model(
        name=config.model,
        device=config.device,
        dtype=config.dtype,
    )

    engine = BatchGraphEngine(
        model=model,
        max_len=config.max_len,
        max_n_beams=1,
        max_batch_size=config.batch_size,
        use_cuda_graph=False,
    )

    logger.info("Initializing tokenizer %s", config.model)
    tokenizer = util.load_tokenizer(config.model)

    logger.info("Initializing dataset %s", config.dataset)
    dataset = util.load_dataset(config.dataset, model_name=config.model)[
        : config.max_request_num
    ]

    curr_req_idx = 0
    req_indices: list[Optional[int]] = [None for _ in range(config.batch_size)]

    engine.reset()
    input_ids = torch.zeros(
        (config.batch_size, 1), dtype=torch.long, device=config.device
    )
    position_ids = torch.zeros(
        (config.batch_size, 1), dtype=torch.long, device=config.device
    )
    cache_batch_indices = torch.arange(
        config.batch_size, dtype=torch.long, device=config.device
    )
    cache_seq_indices = torch.zeros(
        config.batch_size, dtype=torch.long, device=config.device
    )
    attention_mask = torch.zeros(
        (config.batch_size, 1, 1, config.max_len),
        dtype=config.dtype,
        device=config.device,
    )
    offset = torch.zeros(config.batch_size, dtype=torch.long, device=config.device)
    generated_cnt = torch.zeros(
        config.batch_size, dtype=torch.long, device=config.device
    )
    generated_tokens = torch.zeros(
        (config.batch_size, config.max_new_tokens),
        dtype=torch.long,
        device=config.device,
    )

    server_step_idx = 0

    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Processing requests...", total=len(dataset) // config.sample_req_cnt
        )
        while True:
            prefill_cnt = 0
            with util.Timing(device=config.device, mode="sync") as cycle_t:
                for batch_idx, req_idx in enumerate(req_indices):
                    if curr_req_idx >= len(dataset):
                        break

                    if req_idx is not None:
                        continue

                    logger.debug("Request %d is being processed.", curr_req_idx)
                    req_indices[batch_idx] = curr_req_idx
                    prompt = dataset[curr_req_idx]
                    last_token, _offset = _prefill(tokenizer, engine, prompt, batch_idx)

                    input_ids[batch_idx] = last_token
                    position_ids[batch_idx] = _offset
                    cache_seq_indices[batch_idx] = _offset
                    attention_mask[batch_idx, 0, 0, : _offset + 1] = 1.0
                    offset[batch_idx] = _offset
                    curr_req_idx += config.sample_req_cnt
                    prefill_cnt += 1
                    progress.update(task, advance=1)

                logits = engine.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    cache_batch_indices=cache_batch_indices,
                    cache_seq_indices=cache_seq_indices,
                    attention_mask=attention_mask,
                )

                input_ids = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
                position_ids.add_(1)
                cache_seq_indices.add_(1)
                offset.add_(1)

                generated_tokens.scatter_(1, generated_cnt.unsqueeze(1), input_ids)
                generated_cnt.add_(1)
                attention_mask[:, 0, 0, offset] = 1.0

            for batch_idx, req_idx in enumerate(req_indices):
                if req_idx is None:
                    continue

                result_logger.log(
                    {
                        "client_idx": 0,
                        "req_idx": req_idx,
                        "step_idx": generated_cnt[batch_idx].item() - 1,
                        "server_step_idx": server_step_idx,
                        "forward_t": cycle_t.elapsed,
                        "prefill_cnt": prefill_cnt,
                    }
                )

                if (
                    generated_cnt[batch_idx] >= config.max_new_tokens
                    or input_ids[batch_idx] == tokenizer.eos_token_id
                ):
                    if input_ids[batch_idx] == tokenizer.eos_token_id:
                        logger.debug("Request %d completed with EOS token.", req_idx)

                    logger.debug("Request %d completed.", req_idx)
                    logger.debug(
                        "Generated tokens: %s",
                        tokenizer.decode(generated_tokens[batch_idx]),
                    )

                    input_ids[batch_idx] = 0
                    position_ids[batch_idx] = 0
                    cache_seq_indices[batch_idx] = 0
                    attention_mask[batch_idx].zero_()
                    offset[batch_idx] = 0
                    generated_cnt[batch_idx] = 0
                    generated_tokens[batch_idx].zero_()

                    req_indices[batch_idx] = None

                    if (
                        req_idx
                        >= ((len(dataset) - 1) // config.sample_req_cnt)
                        * config.sample_req_cnt
                    ):
                        logger.info("All requests completed.")
                        return

            server_step_idx += 1


def _prefill(tokenizer, engine: BatchGraphEngine, prompt: str, batch_idx: int):
    _input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
    _position_ids = torch.arange(
        _input_ids.shape[-1], dtype=torch.long, device=config.device
    ).unsqueeze(0)
    _cache_seq_indices = torch.arange(
        _input_ids.shape[-1], dtype=torch.long, device=config.device
    )
    _attention_mask = torch.ones(
        (1, 1, _input_ids.shape[-1], config.max_len),
        dtype=config.dtype,
        device=config.device,
    ).tril_()

    engine.prefill(
        input_ids=_input_ids,
        position_ids=_position_ids,
        cache_seq_indices=_cache_seq_indices,
        attention_mask=_attention_mask,
        batch_idx=batch_idx,
    )

    last_token = _input_ids[0, -1]
    offset = _input_ids.shape[-1] - 1

    return last_token, offset


def _open_config(config_file: Path):
    with open(config_file, "r") as f:
        config_yaml = yaml.safe_load(f)

    try:
        git_commit_hash = subprocess.check_output(  # noqa: S603
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            text=True,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        git_commit_hash = "unknown"

    base_config = config_yaml
    base_config["commit"] = git_commit_hash

    return base_config


def _load_config(config: dict):
    entire_result_path = (
        Path(config["base"]["result_path"]) / config["base"]["exp_name"]
    )
    entire_result_path.mkdir(parents=True, exist_ok=True)

    config_file_path = (
        Path(config["base"]["result_path"])
        / config["base"]["exp_name"]
        / f"{config['base']['exp_name']}.yaml"
    )

    with open(config_file_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    os.environ["SPECEDGE_RESULT_PATH"] = config["base"]["result_path"]
    os.environ["SPECEDGE_EXP_NAME"] = config["base"]["exp_name"]
    os.environ["SPECEDGE_PROCESS_NAME"] = "auto"
    os.environ["SPECEDGE_SEED"] = str(config["base"]["seed"])
    os.environ["SPECEDGE_MODEL"] = config["base"]["model"]
    os.environ["SPECEDGE_DEVICE"] = config["base"]["device"]
    os.environ["SPECEDGE_DTYPE"] = config["base"]["dtype"]
    os.environ["SPECEDGE_TEMPERATURE"] = str(config["base"]["temperature"])
    os.environ["SPECEDGE_DATASET"] = config["base"]["dataset"]
    os.environ["SPECEDGE_BATCH_SIZE"] = str(config["base"]["batch_size"])
    os.environ["SPECEDGE_MAX_LEN"] = str(config["base"]["max_len"])
    os.environ["SPECEDGE_MAX_NEW_TOKENS"] = str(config["base"]["max_new_tokens"])
    os.environ["SPECEDGE_MAX_REQUEST_NUM"] = str(config["base"]["max_request_num"])
    os.environ["SPECEDGE_SAMPLE_REQ_CNT"] = str(config["base"]["sample_req_cnt"])

    log_dir = Path(config["base"]["result_path"]) / config["base"]["exp_name"]
    log_config = log.get_default_log_config(log_dir, "auto")
    log.configure_logging(log_config)
    log.log_unexpected_exception()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    util.warn_not_commit()

    config_file = Path(args.config)

    if not config_file.exists() or not config_file.is_file():
        raise FileNotFoundError(f"Cannot open {config_file}.")

    base_config = _open_config(config_file)
    _load_config(base_config)

    main()
