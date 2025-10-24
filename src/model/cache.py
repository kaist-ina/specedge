from contextlib import contextmanager
from typing import Optional

import torch
from transformers.models.llama.configuration_llama import LlamaConfig


class KVCache:
    """
    KVCache is a key-value cache that stores the key-value pairs in the memory.
    It is used to store KV cache in custom llama model
    due to CUDA Graph capture limitation.
    """

    def __init__(
        self,
        config: LlamaConfig,
        max_n_beams: int,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
    ) -> None:
        self.model_config = config
        self.max_len = max_len
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size

        self.num_layers = self.model_config.num_hidden_layers
        self.head_dim = getattr(
            config,
            "head_dim",
            self.model_config.hidden_size // self.model_config.num_attention_heads,
        )
        self._offset = torch.zeros(
            self.batch_size, device=self.device, dtype=torch.long
        )

        self.seq_indices = torch.arange(
            max_n_beams, device=self.device, dtype=torch.long
        ).repeat(batch_size)

        self.k_cache = torch.zeros(
            (
                self.num_layers,
                self.batch_size,
                self.model_config.num_key_value_heads,
                self.max_len,
                self.head_dim,
            ),
            device=self.device,
            dtype=self.dtype,
        )

        self.v_cache = torch.zeros(
            (
                self.num_layers,
                self.batch_size,
                self.model_config.num_key_value_heads,
                self.max_len,
                self.head_dim,
            ),
            device=self.device,
            dtype=self.dtype,
        )

    def gather(
        self, batch_idx: int, src_indices: torch.Tensor, dest_indices: torch.Tensor
    ):
        """
        Remove the key-value pairs that are not used in the current batch.
        """
        src_indices = src_indices.to(self.device)
        dest_indices = dest_indices.to(self.device)

        if src_indices.dtype == torch.bool:
            src_indices = torch.where(src_indices)[0]

        self.k_cache[:, batch_idx, :, dest_indices, :] = self.k_cache[
            :, batch_idx, :, src_indices, :
        ]
        self.v_cache[:, batch_idx, :, dest_indices, :] = self.v_cache[
            :, batch_idx, :, src_indices, :
        ]

        self.k_cache[:, batch_idx, :, dest_indices.max() + 1 :, :].zero_()
        self.v_cache[:, batch_idx, :, dest_indices.max() + 1 :, :].zero_()

    def update(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        layer_idx: int,
        cache_batch_indices: torch.Tensor,
        cache_seq_indices: torch.Tensor,
    ):
        """
        Update the cache with the given key and value tensors.
        """
        self.k_cache[layer_idx, cache_batch_indices, :, cache_seq_indices, :] = k_cache[
            cache_batch_indices, :, self.seq_indices, :
        ]
        self.v_cache[layer_idx, cache_batch_indices, :, cache_seq_indices, :] = v_cache[
            cache_batch_indices, :, self.seq_indices, :
        ]

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def clear(self, batch_indices: Optional[torch.Tensor] = None):
        if batch_indices is not None:
            batch_indices = batch_indices.to(self.device)
            self.k_cache[:, batch_indices, ...].zero_()
            self.v_cache[:, batch_indices, ...].zero_()
            self._offset[batch_indices].zero_()
        else:
            self.k_cache.zero_()
            self.v_cache.zero_()

    @contextmanager
    def prefill_context(self, n_beams: int, batch_idx: int):
        prev_seq_indices = self.seq_indices
        prev_k_cache = self.k_cache
        prev_v_cache = self.v_cache

        self.seq_indices = torch.arange(n_beams, device=self.device)
        self.k_cache = prev_k_cache.select(1, batch_idx).clone().unsqueeze(1)
        self.v_cache = prev_v_cache.select(1, batch_idx).clone().unsqueeze(1)

        yield

        prev_k_cache[:, batch_idx : batch_idx + 1] = self.k_cache
        prev_v_cache[:, batch_idx : batch_idx + 1] = self.v_cache

        self.seq_indices = prev_seq_indices
        self.k_cache = prev_k_cache
        self.v_cache = prev_v_cache
