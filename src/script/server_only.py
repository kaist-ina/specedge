import argparse
import asyncio
import os
import random
from pathlib import Path

import yaml
from rich.progress import Progress

import log
import util
from config import SpecEdgeBatchClientConfig as client_config
from config import SpecEdgeBatchServerConfig as server_config
from specedge.engine.graph import BatchGraphEngine
from specedge.tree import BatchTree
from strategy.edge_draft.specexec import SpecExecEdgeDraft
from strategy.edge_verify.specexec import SpecExecEdgeVerify
from strategy.request_manager import RequestManager
from strategy.server_verify.specexec.server_only import SpecExecServerVerify


async def main():
    logger = log.get_logger()
    result_logger = log.get_result_logger()

    logger.info("Starting SpecEdge edge server...")

    logger.info("Initializing dataset %s...", client_config.dataset)
    dataset = util.load_dataset(
        client_config.dataset, model_name=client_config.draft_model
    )

    dataset_indices = list(range(len(dataset)))[server_config.req_offset :]

    if client_config.max_request_num > 0:
        dataset_indices = dataset_indices[: client_config.max_request_num]
    elif client_config.max_request_num != -1:
        raise ValueError(
            f"Invalid max_request_num: {client_config.max_request_num}. "
            "It should be either -1 or a positive integer."
        )

    dataset_indices = dataset_indices[:: client_config.sample_req_cnt]

    random.seed(0)
    random.shuffle(dataset_indices)

    logger.info("Initializing tree...")
    tree = BatchTree(
        device=client_config.device,
        dtype=client_config.dtype,
        max_len=client_config.max_len,
        batch_size=client_config.max_batch_size,
    )

    logger.info("Initializing tokenizer...")
    _tokenizer = util.load_tokenizer(client_config.draft_model)

    logger.info("Initializing request manager...")
    req_manager = RequestManager(
        max_batch_size=client_config.max_batch_size,
        device=client_config.device,
    )

    logger.info("Initializing draft model...")
    draft_model = util.load_graph_model(
        name=client_config.draft_model,
        device=client_config.device,
        dtype=client_config.dtype,
    )

    logger.info("Initializing target model...")
    target_model = util.load_graph_model(
        name=server_config.target_model,
        device=server_config.device,
        dtype=server_config.dtype,
    )

    logger.info("Initializing Draft Engine...")
    draft_engine = BatchGraphEngine(
        model=draft_model,
        max_len=client_config.max_len,
        max_batch_size=client_config.max_batch_size,
        max_n_beams=client_config.max_n_beams,
    )

    logger.info("Initializing Target Engine...")
    target_engine = BatchGraphEngine(
        model=target_model,
        max_len=client_config.max_len,
        max_batch_size=client_config.max_batch_size,
        max_n_beams=client_config.max_budget + 1,
        use_cuda_graph=False,
    )

    logger.info("Initializing SpecEdge Edge-Draft...")
    edge_draft = SpecExecEdgeDraft(
        tree=tree,
        dataset=dataset,
        dataset_indices=dataset_indices,
        engine=draft_engine,
        req_manager=req_manager,
    )

    logger.info("Initializing SpecEdge Edge-Verify...")
    edge_verify = SpecExecEdgeVerify(
        tree=tree,
        eos_token=_tokenizer.eos_token_id,
        draft_engine=draft_engine,
        target_engine=target_engine,
        req_manager=req_manager,
    )

    logger.info("Initializing SpecEdge Server-Verify...")
    server_verify = SpecExecServerVerify(engine=target_engine)

    iter_idx = 0

    with Progress() as progress:
        task = progress.add_task("Benchmark", total=len(dataset_indices))
        while True:
            logger.debug("iter_idx=%s", iter_idx)

            prev_progress = edge_draft._current_req_idx

            with util.Timing(device=client_config.device, mode="sync") as draft_t:
                prefill_requests, exhausted = await edge_draft.draft(iter_idx)

                if exhausted:
                    break

            progress.update(task, advance=edge_draft._current_req_idx - prev_progress)

            with util.Timing(device=server_config.device, mode="sync") as target_t:
                (
                    input_ids,
                    position_ids,
                    batch_indices,
                    cache_batch_indices,
                    cache_seq_indices,
                    attention_mask,
                    adjusted_seq_indices,
                    original_seq_indices,
                ) = await edge_verify.edge_pre_verify()

                selection = server_verify.server_verify(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    cache_batch_indices=cache_batch_indices,
                    cache_seq_indices=cache_seq_indices,
                    attention_mask=attention_mask,
                    prefill_requests=prefill_requests,
                )

                n_fresh_tokens = await edge_verify.edge_post_verify(
                    selection=selection,
                    batch_indices=batch_indices,
                    adjusted_seq_indices=adjusted_seq_indices,
                    original_seq_indices=original_seq_indices,
                )

            n_fresh_tokens = n_fresh_tokens.cpu().numpy()
            for batch_idx, req_status in enumerate(req_manager.req_statuses):
                if not req_status:
                    continue

                result_logger.log(
                    {
                        "client_idx": 0,
                        "req_idx": req_status.req_idx,
                        "iter_idx": iter_idx - req_status.iter_idx,
                        "server_iter_idx": iter_idx,
                        "draft": {
                            "end_to_end": round(draft_t.elapsed, 4),
                        },
                        "target": {
                            "end_to_end": round(target_t.elapsed, 4),
                        },
                        "num_accepted_tokens": int(n_fresh_tokens[batch_idx]),
                        "prefill": len(prefill_requests),
                    }
                )

            iter_idx += 1


def _load_config(config_file: Path):
    with open(config_file, "r") as f:
        config_yaml = yaml.safe_load(f)

    result_path = config_yaml["base"]["result_path"]
    exp_name = config_yaml["base"]["exp_name"]
    client_device = config_yaml["client"]["device"]
    server_device = config_yaml["server"]["device"]
    dtype = config_yaml["base"]["dtype"]
    seed = config_yaml["base"]["seed"]
    max_len = config_yaml["base"]["max_len"]

    log_config = log.get_default_log_config(Path(result_path) / exp_name, "server_only")
    log.configure_logging(log_config)
    log.log_unexpected_exception()

    logger = log.get_logger()
    logger.debug("result_path: %s", result_path)
    logger.debug("exp_name: %s", exp_name)
    logger.debug("dtype: %s", dtype)
    logger.debug("seed: %s", seed)
    logger.debug("max_len: %s", max_len)
    logger.debug("client_device: %s", client_device)
    logger.debug("server_device: %s", server_device)

    os.environ["SPECEDGE_RESULT_PATH"] = result_path
    os.environ["SPECEDGE_EXP_NAME"] = exp_name
    os.environ["SPECEDGE_PROCESS_NAME"] = "edge"
    os.environ["SPECEDGE_CLIENT_DEVICE"] = client_device
    os.environ["SPECEDGE_SERVER_DEVICE"] = server_device
    os.environ["SPECEDGE_DTYPE"] = dtype
    os.environ["SPECEDGE_SEED"] = str(seed)
    os.environ["SPECEDGE_MAX_LEN"] = str(max_len)

    # client configuration
    host = config_yaml["client"]["host"]
    base_process_name = config_yaml["client"]["process_name"]
    draft_model = config_yaml["client"]["draft_model"]
    target_model = config_yaml["server"]["target_model"]
    temperature = config_yaml["server"]["temperature"]
    dataset = config_yaml["client"]["dataset"]
    max_n_beams = config_yaml["client"]["max_n_beams"]
    max_beam_len = config_yaml["client"]["max_beam_len"]
    max_branch_width = config_yaml["client"]["max_branch_width"]
    max_budget = config_yaml["client"]["max_budget"]

    logger.debug("host: %s", host)
    logger.debug("base_process_name: %s", base_process_name)
    logger.debug("draft_model: %s", draft_model)
    logger.debug("dataset: %s", dataset)
    logger.debug("max_n_beams: %s", max_n_beams)
    logger.debug("max_beam_len: %s", max_beam_len)
    logger.debug("max_branch_width: %s", max_branch_width)
    logger.debug("max_budget: %s", max_budget)

    os.environ["SPECEDGE_HOST"] = host
    os.environ["SPECEDGE_DRAFT_MODEL"] = draft_model
    os.environ["SPECEDGE_TARGET_MODEL"] = target_model
    os.environ["SPECEDGE_TEMPERATURE"] = str(temperature)
    os.environ["SPECEDGE_DATASET"] = dataset
    os.environ["SPECEDGE_MAX_N_BEAMS"] = str(max_n_beams)
    os.environ["SPECEDGE_MAX_BEAM_LEN"] = str(max_beam_len)
    os.environ["SPECEDGE_MAX_BRANCH_WIDTH"] = str(max_branch_width)
    os.environ["SPECEDGE_MAX_BUDGET"] = str(max_budget)

    max_new_tokens = config_yaml["client"]["max_new_tokens"]
    max_request_num = config_yaml["client"]["max_request_num"]
    max_batch_size = config_yaml["client"]["max_batch_size"]
    sample_req_cnt = config_yaml["client"]["sample_req_cnt"]
    num_clients = 1

    logger.debug("max_new_tokens: %s", max_new_tokens)
    logger.debug("max_request_num: %s", max_request_num)
    logger.debug("max_batch_size: %s", max_batch_size)

    os.environ["SPECEDGE_MAX_NEW_TOKENS"] = str(max_new_tokens)
    os.environ["SPECEDGE_MAX_REQUEST_NUM"] = str(max_request_num)
    os.environ["SPECEDGE_MAX_BATCH_SIZE"] = str(max_batch_size)
    os.environ["SPECEDGE_NUM_CLIENTS"] = str(num_clients)
    os.environ["SPECEDGE_SAMPLE_REQ_CNT"] = str(sample_req_cnt)

    os.environ["SPECEDGE_REQ_IDX"] = "0"
    os.environ["SPECEDGE_BATCH_TYPE"] = ""
    os.environ["SPECEDGE_REQ_OFFSET"] = "0"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    config_file_path = Path(args.config)
    _load_config(config_file_path)
    asyncio.run(main())
