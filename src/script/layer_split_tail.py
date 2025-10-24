import socket
import struct
from pathlib import Path

import torch

import log
import util
from model.layer_split import PartialSecondHalf
from specedge.engine.graph import GraphEngine

MODEL = "Qwen/Qwen3-32B"
HOST = "0.0.0.0"  # noqa: S104
PORT = 8000
MAX_LEN = 2048


def send(conn, selection):
    conn.sendall(struct.pack(">I", selection))


def recv(conn, hidden_size):
    hdr = b""

    while len(hdr) < 4:
        chunk = conn.recv(4 - len(hdr))
        if not chunk:
            raise RuntimeError("Connection closed")
        hdr += chunk
    msg_len = struct.unpack(">I", hdr)[0]

    iter_idx = b""
    while len(iter_idx) < 4:
        chunk = conn.recv(4 - len(iter_idx))
        if not chunk:
            raise RuntimeError("Connection closed")
        iter_idx += chunk
    iter_idx = struct.unpack(">I", iter_idx)[0]

    logit_bytes = b""
    while len(logit_bytes) < msg_len:
        chunk = conn.recv(msg_len - len(logit_bytes))
        if not chunk:
            raise RuntimeError("Connection closed")
        logit_bytes += chunk

    return iter_idx, util.decode(
        logit_bytes,
        device=torch.device("cuda:0"),
        dtype=torch.float16,
        shape=(1, -1, hidden_size),
    )


@torch.inference_mode()
def main():
    logger = log.get_logger()
    result_logger = log.get_result_logger()

    prefill_model = util.load_graph_model(
        MODEL,
        torch.device("cpu"),
        torch.float16,
    )

    full_model = util.load_graph_model(
        MODEL,
        torch.device("cpu"),
        torch.float16,
    )

    num_layers = len(full_model.model.layers)
    half_start = num_layers // 2

    partial_model = PartialSecondHalf(full_model, num_layers, half_start).to("cuda:0")
    hidden_size = full_model.model.config.hidden_size

    engine = GraphEngine(partial_model, MAX_LEN, 1)

    prefill_engine = GraphEngine(prefill_model, MAX_LEN, 1)

    tokenizer = util.load_tokenizer(MODEL)

    dataset = util.load_dataset(
        "specbench",
        MODEL,
    )

    dataset = dataset[::8]

    prompt_idx = -1
    offset = 0

    hidden_state = torch.zeros(
        (1, 1, hidden_size), device="cuda:0", dtype=torch.float16
    )
    position_ids = torch.zeros((1, 1), device="cuda:0", dtype=torch.long)
    cache_seq_indices = torch.zeros((1, 1), device="cuda:0", dtype=torch.long)
    cache_batch_indices = torch.zeros((1, 1), device="cuda:0", dtype=torch.long)
    attention_mask = torch.zeros((1, 1, 1, 2048), device="cuda:0", dtype=torch.float16)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.bind((HOST, PORT))
    sock.listen(1)

    conn, addr = sock.accept()

    try:
        while True:
            try:
                iter_idx, hidden_state = recv(conn, hidden_size)
            except ConnectionError:
                logger.error("Connection closed")
                break

            if iter_idx == 0:
                prompt_idx += 1
                offset = 0

                engine.reset()
                prefill_engine.reset()

                prompt = dataset[prompt_idx]
                prefix_tokens = tokenizer.encode(prompt, return_tensors="pt")

                _position_ids = torch.arange(
                    prefix_tokens.shape[1], dtype=torch.long
                ).unsqueeze(0)
                _cache_seq_indices = torch.arange(
                    prefix_tokens.shape[1], dtype=torch.long
                )
                _attention_mask = torch.ones(
                    (1, 1, prefix_tokens.shape[1], MAX_LEN), dtype=torch.float
                ).tril_()

                prefill_engine.prefill(
                    input_ids=prefix_tokens,
                    position_ids=_position_ids,
                    cache_seq_indices=_cache_seq_indices,
                    attention_mask=_attention_mask,
                    batch_idx=0,
                )
                engine._past_key_values.k_cache.copy_(
                    prefill_engine._past_key_values.k_cache
                )
                engine._past_key_values.v_cache.copy_(
                    prefill_engine._past_key_values.v_cache
                )

                offset = int(prefix_tokens.shape[1] - 1)

                position_ids[0, 0] = offset
                cache_seq_indices[0, 0] = offset
                attention_mask[..., : offset + 1] = 1.0

            with util.Timing(device=torch.device("cuda:0"), mode="sync") as end_to_end:
                logits = engine.forward(
                    input_ids=hidden_state,
                    position_ids=position_ids,
                    cache_seq_indices=cache_seq_indices,
                    cache_batch_indices=cache_batch_indices,
                    attention_mask=attention_mask,
                )

                selection = util.sampler_from_logits(
                    logits,
                    temperature=0.7,
                ).item()

                position_ids.add_(1)
                cache_seq_indices.add_(1)
                attention_mask[..., offset + 1] = 1.0
                offset += 1

            send(conn, selection)

            result_logger.log(
                {
                    "client_idx": 0,
                    "req_idx": prompt_idx,
                    "iter_idx": iter_idx,
                    "tail": end_to_end.elapsed,
                }
            )
    finally:
        conn.close()
        sock.close()


if __name__ == "__main__":
    log_config = log.get_default_log_config(
        Path(f"result/layer_split/{MODEL}"), "layer_split_tail"
    )

    log.configure_logging(log_config)
    log.log_unexpected_exception()

    main()
