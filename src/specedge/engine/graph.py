from typing import Callable

import torch

import log
import util
from model.cache import KVCache


class GraphEngine:
    _warmup_iters = 3

    def __init__(
        self, model, max_len: int, max_n_beams: int, capture_only: bool = False
    ) -> None:
        self._logger = log.get_logger()

        self.max_len = max_len
        self._max_n_beams = max_n_beams

        self._model = model
        self._device = self._model.device
        self._dtype = self._model.dtype
        self._config = self._model.config

        self._past_key_values = KVCache(
            config=self._config,
            batch_size=1,
            max_n_beams=self._max_n_beams,
            max_len=self.max_len,
            device=self._device,
            dtype=self._dtype,
        )
        self._mempool = torch.cuda.graphs.graph_pool_handle()

        self._replays: dict[int, Callable] = {}
        self._capture_graphs(capture_only)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_batch_indices: torch.Tensor,
        cache_seq_indices: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        num_beams = input_ids.size(1)
        attention_mask = util.invert_mask(attention_mask)

        _replay = self._replays.get(num_beams)

        if _replay is not None:
            return _replay(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
            )
        else:
            self._logger.debug("Falling back to non-graph execution")
            return self._model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
                past_key_values=self._past_key_values,
            )[0]

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        batch_idx: int,
        cache_seq_indices: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        attention_mask = util.invert_mask(attention_mask)

        cache_batch_indices = torch.zeros(
            (input_ids.size(-1),), dtype=torch.long, device=self._device
        )

        with self._past_key_values.prefill_context(input_ids.size(-1), batch_idx):
            self._model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
                past_key_values=self._past_key_values,
            )

    def _capture_graphs(self, capture_only: bool = False):
        """Captures CUDA graphs for different numbers of beams.

        Depending on the `capture_only` flag, this method captures graphs for
        either all beam sizes from 1 to `_max_n_beams` or just the graph for
        `_max_n_beams` (if `capture_only` is True and `_max_n_beams > 1`).
        Each captured graph is stored in the `_replays` dictionary, keyed by the
        number of beams. The process involves warming up the model and then
        capturing the forward pass with static tensors. Finally, the KV cache
        is cleared.

        Args:
            capture_only: If True and `_max_n_beams > 1`, only captures
                          the graph for the maximum number of beams.
                          Otherwise, captures graphs for all beam sizes
                          from 1 to `_max_n_beams`. Defaults to False.
        """
        self._logger.debug("Capturing graph")

        if capture_only and self._max_n_beams > 1:
            self._replays[self._max_n_beams] = self._capture_graph(self._max_n_beams)
        else:
            for num_beams in range(1, self._max_n_beams + 1):
                self._replays[num_beams] = self._capture_graph(num_beams)
        self._past_key_values.clear()

    @torch.inference_mode()
    def _capture_graph(self, num_beam: int):
        static_input_ids = torch.zeros(
            (1, num_beam), dtype=torch.long, device=self._device
        )
        static_position_ids = torch.zeros(
            (1, num_beam), dtype=torch.long, device=self._device
        )
        static_cache_batch_indices = torch.full(
            (num_beam,), 0, dtype=torch.long, device=self._device
        )
        static_cache_seq_indices = torch.zeros(
            (num_beam,), dtype=torch.long, device=self._device
        )
        static_attention_mask = torch.zeros(
            (1, 1, num_beam, self.max_len),
            dtype=self._dtype,
            device=self._device,
        )

        with torch.cuda.device(self._device):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())

            tmp_seq_indices = self._past_key_values.seq_indices
            self._past_key_values.seq_indices = tmp_seq_indices[:num_beam]

            with torch.cuda.stream(stream):  # type: ignore
                for _ in range(3):
                    self._model.forward(
                        input_ids=static_input_ids,
                        position_ids=static_position_ids,
                        cache_batch_indices=static_cache_batch_indices,
                        cache_seq_indices=static_cache_seq_indices,
                        attention_mask=static_attention_mask,
                        past_key_values=self._past_key_values,
                    )

                stream.synchronize()

            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(graph, stream=stream, pool=self._mempool):
                static_logits, _ = self._model.forward(
                    input_ids=static_input_ids,
                    position_ids=static_position_ids,
                    cache_batch_indices=static_cache_batch_indices,
                    cache_seq_indices=static_cache_seq_indices,
                    attention_mask=static_attention_mask,
                    past_key_values=self._past_key_values,
                )

            def replay(
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                cache_batch_indices: torch.Tensor,
                cache_seq_indices: torch.Tensor,
                attention_mask: torch.Tensor,
            ):
                static_input_ids.copy_(input_ids)
                static_position_ids.copy_(position_ids)
                static_cache_batch_indices.copy_(cache_batch_indices)
                static_cache_seq_indices.copy_(cache_seq_indices)
                static_attention_mask.copy_(attention_mask)
                graph.replay()

                return static_logits.clone()

            self._past_key_values.seq_indices = tmp_seq_indices

            return replay

    def gather(self, src_indices: torch.Tensor, dest_indices: torch.Tensor):
        self._past_key_values.gather(0, src_indices, dest_indices)

    def reset(self):
        self._past_key_values.clear()


class BatchGraphEngine:
    _warmup_iters = 3

    def __init__(
        self,
        model,
        max_len: int,
        max_batch_size: int,
        max_n_beams: int,
        use_cuda_graph: bool = True,
    ) -> None:
        self._logger = log.get_logger()

        self.max_len = max_len
        self.max_batch_size = max_batch_size
        self._max_n_beams = max_n_beams
        self._use_cuda_graphs = use_cuda_graph

        self._model = model
        self._device = self._model.device
        self._dtype = self._model.dtype
        self._config = self._model.config

        self._past_key_values = KVCache(
            config=self._config,
            max_n_beams=self._max_n_beams,
            max_len=self.max_len,
            device=self._device,
            dtype=self._dtype,
            batch_size=self.max_batch_size,
        )
        self._mempool = torch.cuda.graphs.graph_pool_handle()

        self._replays: dict[int, dict[int, Callable]] = {}

        if self._use_cuda_graphs:
            with torch.cuda.device(self._device):
                self._capture_graphs()

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_batch_indices: torch.Tensor,
        cache_seq_indices: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        batch_size, num_beams = input_ids.shape
        attention_mask = util.invert_mask(attention_mask)

        if self._use_cuda_graphs and (
            _replay := self._replays.get(batch_size, {}).get(num_beams)
        ):
            return _replay(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
            )
        else:
            self._logger.debug("Falling back to non-graph execution")
            return self._model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
                past_key_values=self._past_key_values,
            )[0]

    def _capture_graphs(self, capture_only: bool = False):
        self._logger.debug("Capturing graph")

        self._replays[self.max_batch_size] = {}
        self._replays[self.max_batch_size][self._max_n_beams] = self._capture_graph(
            self.max_batch_size, self._max_n_beams
        )
        self._past_key_values.clear()

    @torch.inference_mode()
    def _capture_graph(self, batch_size: int, num_beam: int):
        # Static tensors are created with max_batch_size
        static_input_ids = torch.zeros(
            (batch_size, num_beam), dtype=torch.long, device=self._device
        )
        static_position_ids = torch.zeros(
            (batch_size, num_beam), dtype=torch.long, device=self._device
        )
        # Assuming cache_position is flattened (batch_size * num_beam)
        static_cache_batch_indices = torch.arange(
            batch_size, device=self._device
        ).repeat_interleave(num_beam)
        static_cache_seq_indices = torch.zeros(
            (batch_size * num_beam,), dtype=torch.long, device=self._device
        )
        static_attention_mask = torch.zeros(
            (batch_size, 1, num_beam, self.max_len),
            dtype=self._dtype,
            device=self._device,
        )

        with torch.cuda.device(self._device):
            stream = torch.cuda.Stream(device=self._device)
            stream.wait_stream(torch.cuda.current_stream(device=self._device))

            with torch.cuda.stream(stream):  # type: ignore
                for _ in range(self._warmup_iters):
                    self._model.forward(
                        input_ids=static_input_ids,
                        position_ids=static_position_ids,
                        cache_batch_indices=static_cache_batch_indices,
                        cache_seq_indices=static_cache_seq_indices,
                        attention_mask=static_attention_mask,
                        past_key_values=self._past_key_values,
                    )

                stream.synchronize()

            torch.cuda.current_stream(device=self._device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(graph, stream=stream, pool=self._mempool):
                static_logits, _ = self._model.forward(
                    input_ids=static_input_ids,
                    position_ids=static_position_ids,
                    cache_batch_indices=static_cache_batch_indices,
                    cache_seq_indices=static_cache_seq_indices,
                    attention_mask=static_attention_mask,
                    past_key_values=self._past_key_values,
                )

            def replay(
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                cache_batch_indices: torch.Tensor,
                cache_seq_indices: torch.Tensor,
                attention_mask: torch.Tensor,
            ):
                """
                Copy inputs into the slice of static tensors
                matching the current batch size
                """
                static_input_ids.copy_(input_ids)
                static_position_ids.copy_(position_ids)
                static_cache_batch_indices.copy_(cache_batch_indices)
                static_cache_seq_indices.copy_(cache_seq_indices)
                static_attention_mask.copy_(attention_mask)

                graph.replay()

                # Return the slice of logits corresponding to the actual batch size
                return static_logits.clone()

            return replay

    def gather(
        self, batch_idx: int, src_indices: torch.Tensor, dest_indices: torch.Tensor
    ):
        self._past_key_values.gather(batch_idx, src_indices, dest_indices)

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        batch_idx: int,
        cache_seq_indices: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        attention_mask = util.invert_mask(attention_mask)

        cache_batch_indices = torch.zeros(
            (input_ids.size(-1),), dtype=torch.long, device=self._device
        )

        with self._past_key_values.prefill_context(input_ids.size(-1), batch_idx):
            self._model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
                past_key_values=self._past_key_values,
            )

    def reset(self):
        self._past_key_values.clear()

    def remove_requests(self, batch_indices: torch.Tensor):
        self._past_key_values.clear(batch_indices)
