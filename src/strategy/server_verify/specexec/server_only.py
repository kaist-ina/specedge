from typing import Optional

import torch

import log
import util
from config import SpecEdgeBatchServerConfig as config
from specedge.engine.graph import BatchGraphEngine


class SpecExecServerVerify:
    def __init__(self, engine: BatchGraphEngine) -> None:
        self._logger = log.get_logger()
        self._result_logger = log.get_logger()

        self._device = config.device
        self._dtype = config.dtype

        self._temperature = config.temperature

        self._max_len = config.max_len
        self._max_batch_size = config.max_batch_size
        self._max_n_beams = config.max_budget + 1

        self._engine = engine
        self._tokenizer = util.load_tokenizer(config.target_model)

    def server_verify(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_batch_indices: torch.Tensor,
        cache_seq_indices: torch.Tensor,
        attention_mask: torch.Tensor,
        prefill_requests: Optional[list[tuple[str, int]]] = None,
    ):
        if prefill_requests:
            for prefix, b_idx in prefill_requests:
                self._target_prefill(prefix, b_idx)

        return self._target_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_batch_indices=cache_batch_indices,
            cache_seq_indices=cache_seq_indices,
            attention_mask=attention_mask,
        )

    def _target_prefill(
        self,
        prefix: str,
        batch_idx: int,
    ):
        prefix_tokens = self._tokenizer.encode(prefix, return_tensors="pt").to(
            self._device
        )

        input_ids = prefix_tokens
        position_ids = torch.arange(input_ids.size(-1), device=self._device).unsqueeze(
            0
        )
        cache_seq_indices = torch.arange(input_ids.size(-1), device=self._device)
        attention_mask = torch.ones(
            (1, 1, input_ids.size(-1), self._max_len),
            dtype=self._dtype,
            device=self._device,
        ).tril_()

        self._engine.prefill(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_seq_indices=cache_seq_indices,
            attention_mask=attention_mask,
            batch_idx=batch_idx,
        )

    def _target_forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_batch_indices: torch.Tensor,
        cache_seq_indices: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        logits = self._engine.forward(
            input_ids=input_ids.to(self._device),
            position_ids=position_ids.to(self._device),
            cache_batch_indices=cache_batch_indices.to(self._device),
            cache_seq_indices=cache_seq_indices.to(self._device),
            attention_mask=attention_mask.to(self._device),
        )

        selection = util.sampler_from_logits(logits, temperature=self._temperature)
        return selection


class SpecExecGrpcServerVerify:
    def __init__(self, engine: BatchGraphEngine) -> None:
        self._logger = log.get_logger()
        self._result_logger = log.get_logger()

        self._device = config.device
        self._dtype = config.dtype

        self._temperature = config.temperature

        self._max_len = config.max_len
        self._max_batch_size = config.max_batch_size
        self._max_n_beams = config.max_budget + 1

        self._engine = engine
        self._tokenizer = util.load_tokenizer(config.target_model)
