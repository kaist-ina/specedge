from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch


class RequestManager:
    def __init__(self, max_batch_size: int, device: torch.device) -> None:
        self._max_batch_size = max_batch_size
        self._device = device

        self.initial_prefix_len = torch.zeros(
            (max_batch_size,), dtype=torch.long, device=self._device
        )
        self.req_statuses: list[Optional[RequestStatus]] = [
            None for _ in range(self._max_batch_size)
        ]

    @property
    def finished(self):
        return all(status is None for status in self.req_statuses)

    def empty_batch_indices(self):
        return [i for i, status in enumerate(self.req_statuses) if status is None]

    def add_request(
        self, prefix_tokens: torch.Tensor, req_idx: int, batch_idx: int, iter_idx: int
    ):
        self.req_statuses[batch_idx] = RequestStatus(
            req_idx=req_idx, phase=RequestPhase.Prefill, iter_idx=iter_idx
        )
        self.initial_prefix_len[batch_idx] = prefix_tokens.numel()

    def remove_requests(self, batch_indices: torch.Tensor):
        self.initial_prefix_len[batch_indices] = 0

        for batch_idx in batch_indices:
            self.req_statuses[batch_idx] = None


class RequestPhase(Enum):
    Prefill = 1
    Generating = 2
    Finished = 3


@dataclass
class RequestStatus:
    req_idx: int
    iter_idx: int
    phase: RequestPhase = field(default=RequestPhase.Prefill)
