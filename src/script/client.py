import asyncio
import random
from pathlib import Path

import grpc

import log
import util
from config import SpecEdgeClientConfig as config
from specedge.client.specexec import SpecExecClient
from specedge.engine.graph import GraphEngine
from specedge_grpc import specedge_pb2, specedge_pb2_grpc


async def main():
    logger = log.get_logger()
    logger.info("Starting SpecEdge Client")

    logger.info(
        "Initializing model %s, %s on %s",
        config.draft_model,
        config.dtype,
        config.device,
    )
    draft_model = util.load_graph_model(
        name=config.draft_model,
        device=config.device,
        dtype=config.dtype,
    )

    logger.info("Initializing engine")
    engine = GraphEngine(
        model=draft_model,
        max_len=config.max_len,
        max_n_beams=config.max_n_beams,
    )

    logger.info("Initializing tokenizer %s", config.draft_model)
    tokenizer = util.load_tokenizer(config.draft_model)

    logger.info("Initializing dataset %s", config.dataset)
    dataset = util.load_dataset(
        config.dataset, model_name=config.draft_model, reasoning=config.reasoning
    )

    max_req_num = (
        len(dataset) if config.max_request_num == -1 else config.max_request_num
    )

    req_indices = list(range(len(dataset)))
    req_indices = req_indices[config.req_offset : max_req_num][:: config.sample_req_cnt]

    random.seed(config.client_idx)
    random.shuffle(req_indices)

    with grpc.insecure_channel(config.host) as channel:
        stub = specedge_pb2_grpc.SpecEdgeServiceStub(channel)
        _ = stub.Sync(specedge_pb2.SyncRequest())

    logger.info("Starting %s requests", config.max_request_num)
    for i, req_idx in enumerate(req_indices):
        logger.info("Request %s/%s, req_idx: %s", i + 1, len(req_indices), req_idx)
        await generate(
            req_idx=req_idx,
            prompt=dataset[req_idx],
            engine=engine,
            tokenizer=tokenizer,
        )


async def generate(engine, tokenizer, req_idx: int, prompt: str):
    client = SpecExecClient(
        engine=engine,
        tokenizer=tokenizer,
        prompt=prompt,
        max_len=config.max_len,
    )

    await client.generate(req_idx)


if __name__ == "__main__":
    log_dir = Path(config.result_path) / config.exp_name
    log_config = log.get_default_log_config(log_dir, config.process_name)
    log.configure_logging(log_config)
    log.log_unexpected_exception()

    asyncio.run(main())
