import asyncio
import multiprocessing as mp
import os
import queue
import threading
from pathlib import Path

import torch
from rich.progress import track

import log
import util
from config import SpecEdgeBatchServerConfig as config
from specedge.engine.graph import BatchGraphEngine
from specedge_grpc import specedge_pb2, specedge_pb2_grpc


class SpecExecBatchServer(specedge_pb2_grpc.SpecEdgeServiceServicer):
    def __init__(
        self,
        shutdown_event: asyncio.Event = None,
    ) -> None:
        self._logger = log.get_logger()

        self._loop = asyncio.get_event_loop()
        self._synced = 0
        self._num_clients = config.num_clients
        self._all_sync = asyncio.Condition()

        self._shutdown_event = shutdown_event
        self._resp_queue_task = None

        self._recv_queue = mp.Queue()
        self._resp_queue = mp.Queue()

        self._resp_futures = {}
        self._resp_lock = threading.Lock()

        self._resp_queue_task = self._loop.create_task(self._init_resp_queue_loop())
        self._init_inference_loop()

    async def _init_resp_queue_loop(self):
        self._logger.debug("Starting response queue loop")
        while True:
            try:
                if self._shutdown_event and self._shutdown_event.is_set():
                    self._logger.info("Response queue loop shutting down...")
                    break

                try:
                    raw_data, client_idx = await self._loop.run_in_executor(
                        None, self._resp_queue.get, True, 0.5  # block=True, timeout=0.5
                    )
                except queue.Empty:
                    continue

                if raw_data is None and client_idx == -1:
                    self._logger.info(
                        "Received shutdown sentinel, stopping response queue loop"
                    )
                    break

                self._logger.debug("Received response for client %d", client_idx)

                with self._resp_lock:
                    if client_idx in self._resp_futures:
                        self._resp_futures[client_idx].set_result(raw_data)
                    else:
                        self._logger.error("Client index not found in futures")
            except Exception as e:
                self._logger.error("Error processing response: %s", e)
                if self._shutdown_event and self._shutdown_event.is_set():
                    break

    async def Sync(self, request, context):
        async with self._all_sync:
            self._synced += 1

            if self._synced == self._num_clients:
                self._synced = 0
                self._all_sync.notify_all()
            else:
                await self._all_sync.wait()

        return specedge_pb2.SyncResponse()

    async def Validate(self, request, context):
        self._logger.info("Received request: %s", request.client_idx)
        fut = asyncio.Future()

        with self._resp_lock:
            self._resp_futures[request.client_idx] = fut

        self._recv_queue.put(request.SerializeToString())
        selection, prefill_cnt = await asyncio.wait_for(fut, timeout=5.0)
        return specedge_pb2.ValidateResponse(selection=selection, prefill=prefill_cnt)

    def _init_inference_loop(self):
        self._inference_process = mp.Process(
            target=_init_inference,
            args=(
                self._num_clients,
                self._recv_queue,
                self._resp_queue,
            ),
            daemon=False,
        )
        self._inference_process.start()

    async def cleanup(self):
        """Clean up resources during shutdown"""
        self._logger.info("Starting cleanup...")

        # Send sentinel to inference process to trigger shutdown
        try:
            self._logger.info("Sending shutdown signal to inference process...")
            self._recv_queue.put(None)
        except Exception as e:
            self._logger.exception("Error sending shutdown signal %s", e)

        # Wait for inference process to finish (with timeout)
        if self._inference_process and self._inference_process.is_alive():
            self._logger.info("Waiting for inference process to terminate...")
            self._inference_process.join(timeout=10.0)

            if self._inference_process.is_alive():
                self._logger.warning("Inference process did not terminate, forcing...")
                self._inference_process.terminate()
                self._inference_process.join(timeout=2.0)

                if self._inference_process.is_alive():
                    self._logger.error("Inference process still alive, killing...")
                    self._inference_process.kill()

        # Send sentinel to response queue to stop the loop
        try:
            self._resp_queue.put((None, -1))
        except Exception as e:
            self._logger.error(f"Error sending sentinel to response queue: {e}")

        # Wait for response queue task to complete
        if self._resp_queue_task and not self._resp_queue_task.done():
            self._logger.info("Waiting for response queue task to complete...")
            try:
                await asyncio.wait_for(self._resp_queue_task, timeout=2.0)
            except asyncio.TimeoutError:
                self._logger.warning("Response queue task did not complete in time")
                self._resp_queue_task.cancel()

        # Close queues
        try:
            self._recv_queue.close()
            self._resp_queue.close()
            self._recv_queue.join_thread()
            self._resp_queue.join_thread()
        except Exception as e:
            self._logger.exception("Error closing queue %s", e)

        self._logger.info("Cleanup complete")


class InferenceController:
    def __init__(
        self,
        num_clients: int,
        recv_queue: mp.Queue,
        resp_queue: mp.Queue,
    ) -> None:
        self._logger = log.get_logger()
        self._result_logger = log.get_result_logger()

        self._dtype = config.dtype
        self._device = config.device

        self._num_clients = num_clients
        self._temperature = config.temperature
        self._batch_size = config.max_batch_size
        self._max_budget = config.max_budget
        self._max_n_beams = self._max_budget + 1
        self._max_len = config.max_len
        self._batch_type = config.batch_type
        self.dataset = util.load_dataset(config.dataset, config.target_model)

        self._request_batches: list[specedge_pb2.ValidateRequest] = []
        self._recv_queue = recv_queue
        self._resp_queue = resp_queue

        self._tokenizer = util.load_tokenizer(config.target_model)

        self._logger.info("Initializing inference controller")

        self._logger.debug("Loading model")
        self._model = util.load_graph_model(
            name=config.target_model,
            device=config.device,
            dtype=config.dtype,
        )

        self._engine = BatchGraphEngine(
            model=self._model,
            max_len=config.max_len,
            max_batch_size=config.max_batch_size,
            max_n_beams=self._max_n_beams,
        )

        self.k_cache = torch.zeros(
            (
                self._model.config.num_hidden_layers,
                self._num_clients,
                self._model.config.num_key_value_heads,
                self._max_len,
                self._model.config.head_dim,
            ),
            dtype=self._dtype,
            device=self._device,
        )

        self.v_cache = torch.zeros_like(
            self.k_cache, dtype=self._dtype, device=self._device
        )

        self._client_indices = torch.zeros(
            (self._batch_size,),
            dtype=torch.long,
            device=self._device,
        )

        self._iter_idx = torch.zeros(
            (self._num_clients,),
            dtype=torch.long,
            device=self._device,
        )

        self._input_ids = torch.zeros(
            (self._batch_size, self._max_n_beams),
            dtype=torch.long,
            device=self._device,
        )

        self._parent_indices = torch.zeros(
            (self._batch_size, self._max_budget), dtype=torch.long, device=self._device
        )

        self._position_ids = torch.zeros(
            (self._batch_size, self._max_n_beams),
            dtype=torch.long,
            device=self._device,
        )

        self._cache_batch_indices = torch.arange(
            self._batch_size, dtype=torch.long, device=self._device
        ).repeat_interleave(self._max_n_beams)

        self._cache_seq_indices = torch.zeros(
            (self._batch_size, self._max_n_beams),
            dtype=torch.long,
            device=self._device,
        )

        self._attention_mask = torch.zeros(
            (self._batch_size, 1, self._max_n_beams, self._max_len),
            dtype=self._dtype,
            device=self._device,
        )

        # Predefined tensors for prefill
        self._predefined_position_ids = torch.arange(
            self._max_len, dtype=torch.long, device=self._device
        ).unsqueeze(0)
        self._predefined_attention_mask = torch.ones(
            (1, 1, self._max_len, self._max_len), dtype=self._dtype, device=self._device
        ).tril_()

        self._kv_prefill_offloading = self._cache_prefill()

        self._logger.debug("Inference controller initialized")

    def _cache_prefill(self):
        # Skip prefill caching if disabled
        if not config.cache_prefill:
            self._logger.info("Prefill caching is disabled - will prefill at runtime")
            return {}

        dataset = util.load_dataset(config.dataset, config.target_model)
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")

        if xdg_cache_home is None:
            xdg_cache_home = os.path.join(os.path.expanduser("~"), ".cache")

        cache_folder_name = f"{config.target_model}_{config.dataset}"
        cache_dir = Path(xdg_cache_home) / "specedge" / cache_folder_name

        kv_prefill_offloading: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        req_indices = list(range(len(dataset)))
        req_indices = req_indices[config.req_offset :][:: config.sample_req_cnt]

        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)

        for req_idx in track(req_indices, description="Prefilling cache"):
            k_cache_file_name = cache_dir / f"{req_idx}_key_cache.pt"
            v_cache_file_name = cache_dir / f"{req_idx}_value_cache.pt"

            if k_cache_file_name.exists() and v_cache_file_name.exists():
                self._logger.debug("Cache files already exist for req_idx=%d", req_idx)
                kv_prefill_offloading[req_idx] = (
                    torch.load(k_cache_file_name, map_location="cpu"),
                    torch.load(v_cache_file_name, map_location="cpu"),
                )
                continue

            prompt = dataset[req_idx]

            self._logger.debug("Creating cache files for req_idx=%d", req_idx)

            input_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(
                self._device
            )[..., :-1]
            position_ids = self._predefined_position_ids[:, : input_ids.size(1)]
            cache_seq_indices = self._predefined_position_ids[:, : input_ids.size(1)]
            attention_mask = self._predefined_attention_mask[
                :, :, : input_ids.size(1), : self._max_len
            ]

            self._engine._past_key_values.clear()

            self._engine.prefill(
                input_ids=input_ids,
                position_ids=position_ids,
                batch_idx=0,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
            )

            k_cache = (
                self._engine._past_key_values.k_cache[
                    :, 0, :, : input_ids.size(-1), ...
                ]
                .squeeze(1)
                .clone()
                .detach()
                .cpu()
            )

            v_cache = (
                self._engine._past_key_values.v_cache[
                    :, 0, :, : input_ids.size(-1), ...
                ]
                .squeeze(1)
                .clone()
                .detach()
                .cpu()
            )

            kv_prefill_offloading[req_idx] = (k_cache, v_cache)

            torch.save(k_cache, k_cache_file_name)
            torch.save(v_cache, v_cache_file_name)

        return kv_prefill_offloading

    def loop(self):
        self._logger.debug("Starting inference loop")
        while True:
            if len(self._request_batches) < self._batch_size:
                while self._check_batch_condition():
                    raw_data = self._recv_queue.get()

                    # Check for sentinel value (shutdown signal)
                    if raw_data is None:
                        self._logger.info("Received shutdown signal in inference loop")
                        self._logger.info(
                            "Processing remaining %d requests before shutdown...",
                            len(self._request_batches),
                        )

                        # Process any remaining requests
                        if len(self._request_batches) > 0:
                            self._logger.info(
                                "Processing final batch of %d requests",
                                len(self._request_batches),
                            )
                            self._client_indices.fill_(-1)

                            with util.Timing(
                                device=self._device, mode="sync"
                            ) as inference_t:
                                forward_t, prefill_indices = self._inference(
                                    self._request_batches[-self._batch_size :]
                                )

                            self._result_logger.log(
                                {
                                    "target": {
                                        "forward_t": forward_t,
                                        "server_end_to_end_t": inference_t.elapsed,
                                        "prefill": len(prefill_indices),
                                    }
                                }
                            )

                        self._logger.info("Inference loop shutting down gracefully")
                        return

                    req = specedge_pb2.ValidateRequest()
                    req.ParseFromString(raw_data)
                    self._request_batches.append(req)

                if len(self._request_batches) == 0:
                    continue

                self._logger.info("Batch size reached: %d", len(self._request_batches))

                self._client_indices.fill_(-1)

                with util.Timing(device=self._device, mode="sync") as inference_t:
                    forward_t, prefill_indices = self._inference(
                        self._request_batches[-self._batch_size :]
                    )
                self._request_batches = self._request_batches[: -self._batch_size]

                self._result_logger.log(
                    {
                        "target": {
                            "forward_t": forward_t,
                            "server_end_to_end_t": inference_t.elapsed,
                            "prefill": len(prefill_indices),
                        }
                    }
                )

    @torch.inference_mode()
    def _inference(self, batch: list[specedge_pb2.ValidateRequest]):
        prefill_indices: list[tuple[int, int]] = []
        self._engine._past_key_values.clear()

        for batch_idx, req in enumerate(batch):
            client_idx = req.client_idx
            self._client_indices[batch_idx] = client_idx

            if req.prefill:
                prefill_indices.append((batch_idx, req.req_idx))
                self._iter_idx[req.client_idx] = 0
            else:
                self._iter_idx[req.client_idx] += 1

            self._input_ids[batch_idx].copy_(
                util.decode(req.input_ids, self._device, torch.long, (-1,))
            )
            self._position_ids[batch_idx].copy_(
                util.decode(req.position_ids, self._device, torch.long, (-1,))
            )
            self._parent_indices[batch_idx].copy_(
                util.decode(req.parent_indices, self._device, torch.long, (-1,))
            )
            self._cache_seq_indices[batch_idx].copy_(
                util.decode(req.cache_seq_indices, self._device, torch.long, (-1,))
            )
            self._attention_mask[batch_idx].copy_(
                util.decode(
                    req.attention_mask,
                    self._device,
                    self._dtype,
                    (1, -1, self._max_len),
                )
            )

            if not req.prefill:
                self._engine._past_key_values.k_cache[:, batch_idx, ...].copy_(
                    self.k_cache[:, req.client_idx, ...]
                )
                self._engine._past_key_values.v_cache[:, batch_idx, ...].copy_(
                    self.v_cache[:, req.client_idx, ...]
                )

        for batch_idx, req_idx in prefill_indices:
            if config.cache_prefill:
                # Load from cache
                k_cache, v_cache = self._kv_prefill_offloading[req_idx]

                self._engine._past_key_values.k_cache[
                    :, batch_idx, :, : k_cache.size(2), :
                ].copy_(k_cache)
                self._engine._past_key_values.v_cache[
                    :, batch_idx, :, : v_cache.size(2), :
                ].copy_(v_cache)
            else:
                # Perform runtime prefill
                req = batch[batch_idx]
                if req.prefix is None or req.prefix == "":
                    raise ValueError(
                        f"Prefix is required for runtime prefill (req_idx={req_idx})"
                    )

                input_ids = self._tokenizer.encode(req.prefix, return_tensors="pt").to(
                    self._device
                )[..., :-1]
                position_ids = self._predefined_position_ids[:, : input_ids.size(1)]
                cache_seq_indices = self._predefined_position_ids[
                    :, : input_ids.size(1)
                ]
                attention_mask = self._predefined_attention_mask[
                    :, :, : input_ids.size(1), : self._max_len
                ]

                self._engine.prefill(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    batch_idx=batch_idx,
                    cache_seq_indices=cache_seq_indices,
                    attention_mask=attention_mask,
                )

        with util.Timing(device=self._device, mode="event") as forward_t:
            logits = self._engine.forward(
                input_ids=self._input_ids,
                position_ids=self._position_ids,
                cache_batch_indices=self._cache_batch_indices.flatten(),
                cache_seq_indices=self._cache_seq_indices.flatten(),
                attention_mask=self._attention_mask,
            )

        selection = util.sampler_from_logits(logits, temperature=self._temperature)
        for batch_idx, client_idx in enumerate(self._client_indices):
            if client_idx == -1:
                continue
            self._resp_queue.put(
                (
                    (util.encode(selection[batch_idx]), len(prefill_indices)),
                    client_idx.item(),
                )
            )

        self._reorder_kv_cache(selection=selection)
        return forward_t.elapsed, prefill_indices

    def _check_batch_condition(self):
        match self._batch_type:
            case "dynamic":
                return (
                    self._recv_queue.qsize() > 0
                    and len(self._request_batches) < self._batch_size
                )
            case "static":
                return len(self._request_batches) < self._batch_size
            case _:
                raise ValueError(f"Unknown batch type: {self._batch_type}")

    def _reorder_kv_cache(self, selection: torch.Tensor):
        offset = self._cache_seq_indices[:, 0][None, :].T

        target_choices_list = []
        for batch_idx in range(self._batch_size):
            offset_b = self._cache_seq_indices[batch_idx, 0]
            parent_indices_b = self._parent_indices[batch_idx] - offset_b
            target_choices_b = selection[batch_idx].flatten()[parent_indices_b]
            target_choices_list.append(target_choices_b)
        target_choices = torch.stack(target_choices_list)

        logit_mask = target_choices == self._input_ids[..., 1:]

        _batch_indices = self._cache_batch_indices.flatten()
        _seq_indices = self._cache_seq_indices.flatten()

        tree_mask = torch.empty(
            (self._batch_size, self._max_budget, self._max_budget),
            dtype=torch.float16,
            device=self._device,
        )

        for batch_idx in range(self._batch_size):
            b_offset = self._cache_seq_indices[batch_idx, 1]
            tree_mask[batch_idx].copy_(
                self._attention_mask[
                    batch_idx, 0, 1:, b_offset : b_offset + self._max_budget
                ]
            )

        position = self._position_ids[:, 1:] - offset

        accepted_mask = logit_mask[:, None, :] & tree_mask.to(torch.bool)

        last_accepted_val, last_accepted_indices = (
            position * (accepted_mask.sum(dim=-1) == position)
        ).max(dim=-1)

        last_accepted = torch.where(
            last_accepted_val == 0, 0, last_accepted_indices + 1
        )

        for batch_idx, client_idx in enumerate(self._client_indices):
            if client_idx == -1:
                continue

            src_mask = self._attention_mask[batch_idx, 0, last_accepted[batch_idx], :]
            b_src_indices = torch.where(src_mask)[0]
            b_dest_indices = torch.arange(
                b_src_indices.size(-1), dtype=torch.long, device=self._device
            )
            self._engine.gather(batch_idx, b_src_indices, b_dest_indices)
            self.k_cache[:, client_idx, ...].copy_(
                self._engine._past_key_values.k_cache[:, batch_idx, ...]
            )
            self.v_cache[:, client_idx, ...].copy_(
                self._engine._past_key_values.v_cache[:, batch_idx, ...]
            )


def _init_inference(
    num_clients: int,
    recv_queue: mp.Queue,
    resp_queue: mp.Queue,
):
    # Configure logging in child process
    from config import SpecEdgeBatchServerConfig as config

    log_config = log.get_default_log_config(
        Path(config.result_path) / config.exp_name, "server"
    )
    log.configure_logging(log_config)

    try:
        controller = InferenceController(num_clients, recv_queue, resp_queue)
        controller.loop()
    except KeyboardInterrupt:
        # Gracefully exit without printing traceback
        pass
