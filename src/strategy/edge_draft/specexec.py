import numpy as np
import torch

import log
import util
from config import SpecEdgeBatchClientConfig as config
from specedge.engine.graph import BatchGraphEngine
from specedge.tree import BatchTree
from strategy.request_manager import RequestManager, RequestPhase


class SpecExecEdgeDraft:
    def __init__(
        self,
        tree: BatchTree,
        dataset,
        dataset_indices: list[int],
        engine: BatchGraphEngine,
        req_manager: RequestManager,
    ) -> None:
        self._logger = log.get_logger()
        self._result_logger = log.get_result_logger()

        # Load configuration
        self._device = config.device
        self._dtype = config.dtype

        self._max_len = config.max_len
        self._max_n_beams = config.max_n_beams
        self._max_beam_len = config.max_beam_len
        self._max_branch_width = config.max_branch_width
        self._max_batch_size = config.max_batch_size
        self._max_budget = config.max_budget

        # Load engine
        self._engine = engine

        # Initialize tokenizer
        self._tokenizer = util.load_tokenizer(config.draft_model)

        # Handle status of the requests
        self._req_manager = req_manager

        # Request data structure
        self._tree = tree

        # Request source
        self._dataset = dataset
        self._dataset_indices = dataset_indices
        self._current_req_idx = 0
        self._max_request_num = (
            config.max_request_num
            if config.max_request_num != -1
            else len(self._dataset)
        )

        # Draft predefined tensors

        # shape: [batch_size * max_n_beams]
        self._batch_indices = torch.arange(self._max_batch_size).repeat_interleave(
            self._max_n_beams
        )
        self._candidate_indices = torch.zeros(
            (self._max_batch_size * self._max_n_beams),
            dtype=torch.long,
        )
        self._input_ids = torch.zeros(
            (self._max_batch_size, self._max_beam_len),
            dtype=torch.long,
        )
        self._position_ids = torch.zeros(
            (self._max_batch_size, self._max_beam_len),
            dtype=torch.long,
        )
        self._cache_seq_indices = torch.zeros(
            (self._max_batch_size, self._max_beam_len),
            dtype=torch.long,
        )
        self._attention_mask = torch.zeros(
            (self._max_batch_size, 1, self._max_n_beams, self._max_len),
            dtype=torch.bool,
        )
        self._budget_bucket = torch.full(
            (
                self._max_batch_size,
                self._max_budget + self._max_branch_width * self._max_n_beams,
            ),
            torch.finfo(self._tree.logprobs.dtype).min,
            dtype=self._tree.logprobs.dtype,
            device=self._device,
        )

    async def draft(self, iter_idx: int):
        prefill_requests, exhausted = self._prepare_request(iter_idx)

        if not exhausted:
            self._draft()

        return prefill_requests, exhausted

    def _prepare_request(self, iter_idx: int):
        """
        Prepare the request.
        """
        return self._add_next_requests(iter_idx)

    def _add_next_requests(self, iter_idx: int):
        """
        Add next requests to the request queue.
        """
        prefill_requests = []

        for idx in self._req_manager.empty_batch_indices():
            # Check whether dataset is exhausted
            if self._current_req_idx >= len(self._dataset_indices):
                if self._req_manager.finished:
                    self._logger.info("Dataset exhausted.")
                    return prefill_requests, True
                else:
                    break

            req_idx = self._dataset_indices[self._current_req_idx]

            prefix_tokens = self._tokenizer.encode(
                self._dataset[req_idx], return_tensors="pt"
            ).to(config.device)

            self._req_manager.add_request(
                prefix_tokens=prefix_tokens,
                req_idx=req_idx,
                batch_idx=idx,
                iter_idx=iter_idx,
            )

            self._tree.add_request(
                prefix_tokens=prefix_tokens,
                batch_idx=idx,
            )

            prefill_requests.append((self._dataset[req_idx], idx))
            self._logger.debug("Request %d added in %d.", req_idx, idx)
            self._current_req_idx += 1

        return prefill_requests, False

    @torch.inference_mode()
    def _draft(self):
        self._prefill()

        self._grow_tree()

    def _prefill(self):
        """
        Execute the prefill phase if the request is in the prefill phase.
        """
        for b_idx, req_status in enumerate(self._req_manager.req_statuses):
            if req_status is None or req_status.phase != RequestPhase.Prefill:
                continue

            self._logger.debug("Prefilling batch %d", b_idx)

            input_ids = self._tree.tokens[b_idx, : self._tree.prefix_len[b_idx]][
                None, :
            ]
            position_ids = self._tree.positions[b_idx, : self._tree.prefix_len[b_idx]][
                None, :
            ]
            cache_seq_indices = self._tree.positions[
                b_idx, : self._tree.prefix_len[b_idx]
            ]
            attention_mask = self._tree.amask[
                b_idx, :, : self._tree.prefix_len[b_idx], :
            ].unsqueeze(0)

            self._engine.prefill(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
                batch_idx=b_idx,
            )

            req_status.phase = RequestPhase.Generating

    def _grow_tree(self):
        self._reset_predefined_tensors()
        for _ in range(self._max_beam_len):
            self._process_candidates()

            logits = self._draft_forward()

            (
                batch_indices,
                token_ids,
                token_positions,
                parent_indices,
                token_probs,
            ) = self._get_next_beams(logits)

            self._tree.add(
                batch_indices=batch_indices,
                token_ids=token_ids,
                token_positions=token_positions,
                parent_indices=parent_indices,
                logprobs=token_probs,
            )

        self._trim_by_budget()

    def _reset_predefined_tensors(self):
        self._candidate_indices.zero_()
        self._input_ids.zero_()
        self._position_ids.fill_(self._max_len - 1)
        self._cache_seq_indices.fill_(self._max_len - 1)
        self._attention_mask.zero_()
        self._budget_bucket.fill_(torch.finfo(self._tree.logprobs.dtype).min)

    def _process_candidates(self):
        """
        Processes candidate nodes from the tree for the next draft step,
        and selects candidate nodes from the tree for the next draft step.

        - Selects the top-k candidate nodes for each batch based on their log
          probabilities from the `self._tree`.
        - It masks non-candidate nodes and nodes with very low probabilities
          before selecting the top candidates.
        - Finally, it prepares the necessary input tensors (`_candidate_indices`,
          `_input_ids`, `_position_ids`, `_cache_position`, `_attention_mask`)
          using the selected candidate node indices for the subsequent forward
          pass of the draft model.
        """
        # find candidates for all batches
        # shape: [batch_size, max_len]
        candidate_mask = self._tree.status == self._tree.CANDIDATE

        # Get the logprobs of candidates, and mask the rest with -inf
        # shape: [batch_size, max_len]
        candidate_logprobs = torch.where(
            candidate_mask,
            self._tree.logprobs,
            torch.finfo(self._tree.logprobs.dtype).min,
        )

        # Get top-k candidates
        # shape: [batch_size, max_n_beams]
        candidate_logprobs = candidate_logprobs.topk(
            k=self._max_n_beams, dim=-1, sorted=False
        )

        # For values masked with -inf,
        # set indices to self._max_len - 1
        # shape: [batch_size, max_n_beams]
        raw_candidate_indices = torch.where(
            candidate_logprobs.values != torch.finfo(self._tree.logprobs.dtype).min,
            candidate_logprobs.indices,
            self._max_len - 1,
        )

        # shape: [batch_size * max_n_beams]
        self._candidate_indices = raw_candidate_indices.flatten()

        # shape: [batch_size, max_n_beams]
        self._input_ids = self._tree.tokens[
            self._batch_indices, self._candidate_indices
        ].reshape(self._max_batch_size, -1)

        # shape: [batch_size, max_n_beams]
        self._position_ids = self._tree.positions[
            self._batch_indices, self._candidate_indices
        ].reshape(self._max_batch_size, -1)

        # shape: [batch_size * max_n_beams]
        self._cache_seq_indices = raw_candidate_indices.flatten()

        # shape: [batch_size, 1, max_n_beams, max_len]
        self._attention_mask = self._tree.amask[
            self._batch_indices, :, self._candidate_indices, :
        ].reshape(self._max_batch_size, 1, self._max_n_beams, -1)

        self._tree.status[self._batch_indices, self._candidate_indices] = (
            self._tree.PROCESSED
        )

    def _draft_forward(self):
        logits = self._engine.forward(
            input_ids=self._input_ids,
            position_ids=self._position_ids,
            cache_batch_indices=self._batch_indices,
            cache_seq_indices=self._cache_seq_indices,
            attention_mask=self._attention_mask,
        )
        return logits

    def _get_next_beams(self, logits: torch.Tensor):
        # Calculate log probabilities from the logits.
        logprobs = torch.log_softmax(logits, dim=-1)

        # Select up to max_branch_width tokens from one beam
        # shape: [batch_size, n_beams, max_branch_width]
        logprobs_k = logprobs.topk(k=self._max_branch_width, dim=-1, sorted=False)
        leave_ids = logprobs_k.indices
        leave_probs = logprobs_k.values

        # Mask indices that were previously used for padding
        # shape: [batch_size * max_n_beams]
        pad_mask = self._candidate_indices == self._max_len - 1

        # get the cumulative logprobs for each beams
        # shape: [batch_size, n_beams, 1]
        beam_scores = self._tree.logprobs[
            self._batch_indices, self._candidate_indices
        ].flatten()
        beam_scores[pad_mask] = torch.finfo(logits.dtype).min
        beam_scores = beam_scores.view(self._max_batch_size, self._max_n_beams, -1)

        # Sum accumulated probability, DECAY_FACTOR, and logit logprob
        # shape: [batch_size, n_beams * max_branch_width]

        decay_factor = torch.full(
            (self._max_batch_size, self._max_n_beams, 1),
            np.log(0.9),
            device=self._device,
        )
        flat_incoming_probs = (beam_scores + decay_factor + leave_probs).reshape(
            self._max_batch_size, -1
        )

        # Add `flat_incoming_probs` to the right side of `_budget_bucket`
        self._budget_bucket[:, self._max_budget :] = flat_incoming_probs

        # Update the left side of `_budget_bucket` with the topk values
        self._budget_bucket[:, : self._max_budget] = self._budget_bucket.topk(
            k=self._max_budget, sorted=False
        ).values

        # Find the minimum probability in the current budget bucket
        # shape: [batch_size]
        min_prob = torch.max(
            self._budget_bucket[:, : self._max_budget].min(dim=-1).values,
            torch.tensor(-10),
        )

        # shape: [batch_size, n_beams * max_branch_width]
        flat_incoming_ids = leave_ids.reshape(self._max_batch_size, -1)

        # mask for logits threshold
        # shape: [batch_size * n_beams * max_branch_width]
        in_budget_mask = (flat_incoming_probs >= min_prob.unsqueeze(-1)).flatten()

        in_budget_ids = flat_incoming_ids.flatten()[in_budget_mask]
        in_budget_probs = flat_incoming_probs.flatten()[in_budget_mask]
        in_budget_indices = torch.where(in_budget_mask)[0]

        batch_indices = in_budget_indices // (
            self._max_branch_width * self._max_n_beams
        )
        hypo_ids = (
            in_budget_indices % (self._max_n_beams * self._max_branch_width)
        ) // self._max_branch_width

        # shape: [batch_size, max_len]
        parent_indices = self._candidate_indices.reshape(self._max_batch_size, -1)

        # shape: [batch_size * max_n_beams]
        parent_indices = parent_indices[batch_indices, hypo_ids].flatten()

        beam_positions = self._tree.positions[batch_indices, parent_indices] + 1

        return (
            batch_indices,
            in_budget_ids,
            beam_positions,
            parent_indices,
            in_budget_probs,
        )

    def _trim_by_budget(self):
        for b_idx in range(self._max_batch_size):
            if self._tree.end[b_idx] - self._tree.prefix_len[b_idx] <= self._max_budget:
                continue

            src_indices = (
                self._tree.logprobs[
                    b_idx, self._tree.prefix_len[b_idx] : self._tree.end[b_idx]
                ]
                .topk(k=self._max_budget, sorted=False)
                .indices
                + self._tree.prefix_len[b_idx]
            )

            dest_indices = torch.arange(
                self._tree.prefix_len[b_idx].item(),
                self._tree.prefix_len[b_idx].item() + src_indices.size(-1),
                device=self._device,
            )

            self._tree.gather(b_idx, src_indices, dest_indices)
            self._engine.gather(b_idx, src_indices, dest_indices)
