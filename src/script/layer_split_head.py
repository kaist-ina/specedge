import socket
import struct
from pathlib import Path

import torch

import log
import util
from model.layer_split import PartialFirstHalf
from specedge.engine.graph import GraphEngine

MODEL = "Qwen/Qwen3-32B"
HOST = "127.0.0.1"
PORT = 8000
MAX_LEN = 2048


def send(conn, iter_idx, logit):
    logit_bytes = util.encode(logit)

    conn.sendall(struct.pack(">I", len(logit_bytes)))
    conn.sendall(struct.pack(">I", iter_idx))
    conn.sendall(logit_bytes)


def recv(conn, device):
    next_id = b""

    while len(next_id) < 4:
        chunk = conn.recv(4 - len(next_id))
        if not chunk:
            raise RuntimeError("Connection closed")
        next_id += chunk

    return torch.tensor(
        [[struct.unpack(">I", next_id)[0]]], dtype=torch.long, device=device
    )


@torch.inference_mode()
def main():
    logger = log.get_logger()
    result_logger = log.get_result_logger()

    full_model = util.load_graph_model(MODEL, torch.device("cpu"), torch.float16)
    num_layers = len(full_model.model.layers)

    partial_model = PartialFirstHalf(full_model, num_layers // 2).to("cuda:0")
    tokenizer = util.load_tokenizer(MODEL)

    dataset = util.load_dataset("specbench", MODEL)

    dataset = dataset[::8]

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.connect((HOST, PORT))
    sock.settimeout(None)

    engine = GraphEngine(
        partial_model,
        MAX_LEN,
        1,
    )

    for req_idx, prompt in enumerate(dataset):
        engine.reset()

        prefix_tokens = tokenizer.encode(prompt, return_tensors="pt").to("cuda:0")
        _position_ids = torch.arange(
            prefix_tokens.shape[1], dtype=torch.long, device=prefix_tokens.device
        ).unsqueeze(0)
        _cache_seq_indices = torch.arange(
            prefix_tokens.shape[1], dtype=torch.long, device=prefix_tokens.device
        )
        _attention_mask = torch.ones(
            (1, 1, prefix_tokens.shape[-1], MAX_LEN),
            dtype=torch.float16,
            device=prefix_tokens.device,
        ).tril_()

        engine.prefill(
            input_ids=prefix_tokens,
            position_ids=_position_ids,
            cache_seq_indices=_cache_seq_indices,
            attention_mask=_attention_mask,
            batch_idx=0,
        )

        input_ids = prefix_tokens[..., -1].unsqueeze(0)
        offset = prefix_tokens.shape[-1] - 1

        position_ids = torch.tensor(
            [[offset]], dtype=torch.long, device=prefix_tokens.device
        )
        cache_seq_indices = torch.tensor(
            [[offset]], dtype=torch.long, device=prefix_tokens.device
        )
        cache_batch_indices = torch.zeros(
            (1, 1), dtype=torch.long, device=prefix_tokens.device
        )
        attention_mask = torch.zeros(
            (1, 1, 1, MAX_LEN),
            dtype=torch.float16,
            device=prefix_tokens.device,
        )
        attention_mask[..., : offset + 1] = 1.0

        generated_tokens = input_ids.clone()

        for iter_idx in range(64):
            with util.Timing(device="cuda:0", mode="sync") as end_to_end:
                logits = engine.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    cache_seq_indices=cache_seq_indices,
                    cache_batch_indices=cache_batch_indices,
                    attention_mask=attention_mask,
                )

                send(sock, iter_idx, logits)
                next_id = recv(sock, prefix_tokens.device)

                input_ids = next_id[..., -1].unsqueeze(0)
                position_ids.add_(1)
                cache_seq_indices.add_(1)
                offset += 1
                attention_mask[..., offset] = 1.0
                generated_tokens = torch.cat((generated_tokens, input_ids), dim=-1)

            result_logger.log(
                {
                    "client_idx": 0,
                    "req_idx": req_idx,
                    "iter_idx": iter_idx,
                    "end_to_end": end_to_end.elapsed,
                }
            )

        logger.info(
            "Generated tokens: %s", tokenizer.decode(generated_tokens[0].tolist())
        )


if __name__ == "__main__":
    log_config = log.get_default_log_config(
        Path(f"result/layer_split/{MODEL}"),
        "layer_split_head",
    )

    log.configure_logging(log_config)
    log.log_unexpected_exception()
    main()
