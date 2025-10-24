import asyncio
from typing import Optional

import numpy as np
import torch

import log
import util
from config import SpecEdgeClientConfig as config
from specedge.client.proactive import SpecExecProactiveDraft
from specedge.network.grpc import GrpcClientController
from specedge.tree import Tree


class SpecExecClient:
    def __init__(
        self,
        engine,
        tokenizer,
        prompt: str,
        max_len: int,
    ) -> None:
        # logging
        self._logger = log.get_logger()
        self._result_logger = log.get_result_logger()

        self._logger.debug("Initializing SpecExecClient")

        self._optimization = config.optimization
        self._draft_forward_time_mode = (
            "no-sync" if self._optimization >= 2 else "event"
        )
        self._target_time_mode = "no-sync" if self._optimization >= 2 else "sync"

        self._device = config.device
        self._dtype = config.dtype

        self._max_n_beams = config.max_n_beams
        self._max_beam_len = config.max_beam_len
        self._max_branch_width = config.max_branch_width
        self._max_budget = config.max_budget

        self._proactive_type = config.proactive_type

        self._max_new_tokens = config.max_new_tokens
        self._client_idx = config.client_idx

        self._verify_configs()

        self._engine = engine
        self._tokenizer = tokenizer
        self._engine.reset()

        self._prompt = prompt
        self._prefix_tokens = self._tokenizer.encode(prompt, return_tensors="pt").to(
            self._device
        )[: config.max_len]
        self._num_original_tokens = self._prefix_tokens.numel()
        self._max_len = max_len

        self._tree = Tree(
            prefix_tokens=self._prefix_tokens,
            device=self._device,
            dtype=self._dtype,
            max_len=self._engine.max_len,
        )
        self._validator = GrpcClientController(host=config.host, device=self._device)

        self._proactive_client: Optional[SpecExecProactiveDraft] = None
        if self._proactive_type != "disabled":
            self._proactive_client = SpecExecProactiveDraft(
                tree=self._tree,
                engine=self._engine,
                max_len=self._max_len,
            )

            # Whether Proactive Draft was executed in the previous iter
            self._previous_proactive_draft = False

            # Whether Proactive Draft is executed in the current iter
            self._proactive_draft = False

    def _verify_configs(self):
        if self._proactive_type not in ["included", "excluded", "disabled"]:
            raise ValueError(f"Invalid proactive_type: {self._proactive_type}")

    async def generate(self, req_idx: int):
        """
        Generate a sequence using SpecExec up to max_new_tokens.
        """

        self._logger.info("Generating sequence req_idx=%d", req_idx)

        util.set_seed(config.seed)
        step_idx = 0

        # Prefill phase
        self._logger.debug("Prefill phase: req_idx=%d, step_idx=%d", req_idx, step_idx)
        warmup_tokens = await self._cycle(req_idx, step_idx, prefill=True)
        self._prefix_tokens = torch.cat([self._prefix_tokens, warmup_tokens], dim=-1)

        step_idx = 1
        eos_flag = False

        # speculative decoding phase
        while (
            self._prefix_tokens.numel()
            < self._max_new_tokens + self._num_original_tokens + warmup_tokens.numel()
            and not eos_flag
        ):
            self._logger.debug(
                "Speculative Decoding phase: req_idx=%d, step_idx=%d", req_idx, step_idx
            )
            fresh_tokens = await self._cycle(req_idx, step_idx)

            eos_positions = (fresh_tokens == self._tokenizer.eos_token_id).nonzero()
            if eos_positions.numel() > 0:
                eos_idx = eos_positions[0, 0].item()
                fresh_tokens = fresh_tokens[: eos_idx + 1]
                eos_flag = True

            self._prefix_tokens = torch.cat([self._prefix_tokens, fresh_tokens], dim=-1)
            step_idx += 1

        if eos_flag:
            self._logger.debug("EOS token found.")
        else:
            self._logger.debug("Max new tokens reached.")

        self._logger.info("Finished generating sequence req_idx=%d", req_idx)
        self._logger.info(
            "Generated sequence: \n%s",
            self._tokenizer.decode(self._prefix_tokens[0], skip_special_tokens=True),
        )

    async def _cycle(self, req_idx: int, step_idx: int, prefill=False) -> torch.Tensor:
        with util.Timing(device=self._device, mode="sync") as draft_t:
            draft_stats = self._grow_tree(prefill)

        with util.Timing(device=self._device, mode="sync") as target_t:
            fresh_token_ids, target_stats = await self._validate_tree(req_idx, prefill)

        self._result_logger.log(
            {
                "client_idx": self._client_idx,
                "req_idx": req_idx,
                "step_idx": step_idx,
                "draft": {
                    "forward": draft_stats["forward_t"],
                    "end_to_end": draft_t.elapsed,
                },
                "target": {
                    "client_preprocess": target_stats["preprocess_t"],
                    "client_wait": target_stats["wait_t"],
                    "client_postprocess": target_stats["postprocess_t"],
                    "end_to_end": target_t.elapsed,
                    "prefill": target_stats["prefill"],
                    "proactive": target_stats["proactive"],
                    "prev_proactive": target_stats["previous_proactive"],
                },
                "num_accepted_tokens": target_stats["num_accepted_tokens"],
            }
        )

        return fresh_token_ids

    def _grow_tree(self, prefill: bool):
        self._logger.debug("Growing tree")

        # draft forward times
        draft_forward_times = []

        max_beam_len = self._max_beam_len
        if self._proactive_type == "included" and self._proactive_draft:
            max_beam_len = max(0, self._max_beam_len - config.proactive_max_beam_len)

        if torch.where(self._tree.status == self._tree.CANDIDATE)[0].numel() == 0:
            max_beam_len = 0

        for cnt in range(max_beam_len):
            self._logger.debug("Growing tree: %d / %d", cnt, max_beam_len)

            logits, beam_indices, beam_positions, beam_scores, draft_forward_t = (
                self._process_candidates(prefill)
            )
            prefill = False

            draft_forward_times.append(draft_forward_t)

            (
                next_beam_ids,
                next_beam_positions,
                next_beam_indices,
                beam_logprobs,
            ) = self._get_next_beams(
                logits=logits,
                beam_indices=beam_indices,
                beam_positions=beam_positions,
                beam_scores=beam_scores,
            )

            if next_beam_ids.numel() == 0:
                self._logger.debug("No more beams to grow")
                break

            if (
                self._tree.end - self._tree.prefix_len >= self._max_budget
                and not self._check_new_token_in_budget(beam_logprobs)
            ):
                self._logger.debug("Max budget reached. early stopping")
                break

            self._tree.add(
                token_ids=next_beam_ids,
                token_positions=next_beam_positions,
                parent_indices=next_beam_indices,
                logprobs=beam_logprobs,
            )

        if self._tree.end - self._tree.prefix_len >= self._max_budget:
            self._logger.debug("Trimming tree")
            self._trim_by_budget()

        return {"forward_t": draft_forward_times}

    def _process_candidates(self, warmup: bool):
        self._logger.debug("Processing candidates")
        candidate_indices = torch.where(
            self._tree.status[: self._tree.end] == self._tree.CANDIDATE
        )[0]

        if candidate_indices.numel() > self._max_n_beams:
            self._logger.debug("Choosing top %d candidates", self._max_n_beams)
            cumulative_logprobs = self._tree.logprobs[candidate_indices]
            top_k_indices = cumulative_logprobs.topk(
                k=self._max_n_beams, sorted=False
            ).indices
            candidate_indices = candidate_indices[top_k_indices]
            candidate_indices, _ = candidate_indices.sort()

        if warmup:
            prefill_input_indices = torch.arange(
                candidate_indices.min().item(), device=self._device
            )
            prefill_input_ids = self._tree.tokens[prefill_input_indices].unsqueeze(0)
            prefill_position_ids = self._tree.positions[
                prefill_input_indices
            ].unsqueeze(0)
            prefill_cache_seq_indices = prefill_input_indices
            prefill_attention_mask = self._tree.amask[..., prefill_input_indices, :]

            self._engine.prefill(
                input_ids=prefill_input_ids,
                position_ids=prefill_position_ids,
                batch_idx=0,
                cache_seq_indices=prefill_cache_seq_indices,
                attention_mask=prefill_attention_mask,
            )

        input_indices = candidate_indices

        input_ids = self._tree.tokens[input_indices].unsqueeze(0)
        position_ids = self._tree.positions[input_indices].unsqueeze(0)
        cache_seq_indices = input_indices
        cache_batch_indices = torch.full_like(
            cache_seq_indices, 0, dtype=torch.long, device=self._device
        )
        attention_mask = self._tree.amask[..., input_indices, :]

        with util.Timing(device=self._device, mode=self._draft_forward_time_mode) as t:
            logits = self._engine.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
            )

        self._tree.status[candidate_indices] = self._tree.PROCESSED
        beam_scores = self._tree.logprobs[candidate_indices]
        beam_positions = self._tree.positions[candidate_indices]
        logits = logits[0, -candidate_indices.size(-1) :, :]

        return (logits, candidate_indices, beam_positions, beam_scores, t.elapsed)

    def _get_next_beams(
        self,
        logits: torch.Tensor,
        beam_indices: torch.Tensor,
        beam_positions: torch.Tensor,
        beam_scores: torch.Tensor,
    ):
        self._logger.debug("Getting next beams")
        DECAY_FACTOR = np.log(0.9)

        logprobs = torch.log_softmax(logits, dim=-1)  # shape: [n_beams, vocab_size]
        logprobs_k = logprobs.topk(
            k=self._max_branch_width, dim=-1, sorted=False
        )  # shape: [n_beams, max_branch_width]
        leaves_ids = logprobs_k.indices
        leaves_probs = logprobs_k.values

        flat_incoming_probs = (
            beam_scores.unsqueeze(-1) + DECAY_FACTOR + leaves_probs
        ).flatten()
        flat_incoming_ids = leaves_ids.flatten()

        joint_probs = torch.concat(
            [
                self._tree.logprobs[self._tree.prefix_len : self._tree.end],
                flat_incoming_probs,
            ]
        )

        if (
            joint_probs.size(-1) > self._max_budget
            or joint_probs.size(-1) + (self._tree.end - self._tree.prefix_len)
            > self._max_len
        ):
            min_joint_prob = joint_probs.topk(
                k=self._max_budget, sorted=False, dim=-1
            ).values.min()

            flat_best_mask = torch.where(flat_incoming_probs >= min_joint_prob)[0]
            flat_best_probs = flat_incoming_probs[flat_best_mask]
            flat_best_indices = flat_best_mask
            best_children_token_ids = flat_incoming_ids[flat_best_indices]

            if flat_best_indices.size(-1) + self._tree.end > self._max_len:
                raise NotImplementedError("Implement trim budget")

        else:
            flat_best_probs = flat_incoming_probs
            flat_best_indices = torch.arange(
                flat_incoming_probs.size(0), device=logits.device
            )
            best_children_token_ids = flat_incoming_ids

        best_hypo_ids = flat_best_indices // self._max_branch_width
        best_beam_indices = beam_indices[best_hypo_ids]
        best_children_positions = beam_positions[best_hypo_ids] + 1

        return (
            best_children_token_ids,
            best_children_positions,
            best_beam_indices,
            flat_best_probs,
        )

    def _check_new_token_in_budget(self, cumulative_beam_scores: torch.Tensor):
        lowest_tree_logprob = (
            self._tree.logprobs[self._tree.prefix_len : self._tree.end]
            .topk(k=self._max_budget, dim=-1, sorted=False)
            .values.min()
        )
        best_new_logprob = cumulative_beam_scores.max()

        return best_new_logprob >= lowest_tree_logprob

    def _trim_by_budget(self):
        src_indices = (
            self._tree.logprobs[self._tree.prefix_len : self._tree.end]
            .topk(k=self._max_budget, sorted=False)
            .indices
            + self._tree.prefix_len
        )
        dest_indices = torch.arange(
            self._tree.prefix_len,
            self._tree.prefix_len + src_indices.size(-1),
            device=self._device,
        )

        self._tree.gather(src_indices, dest_indices)
        self._engine.gather(src_indices, dest_indices)

    async def _validate_tree(self, req_idx: int, prefill=False):
        self._logger.debug("Validating tree")

        with util.Timing(
            device=self._device, mode=self._target_time_mode
        ) as preprocess_t:
            target_token_map_bool = (
                self._tree.status[: self._tree.end] >= self._tree.PROCESSED
            )
            target_token_map_bool[: self._tree.prefix_len] = False
            target_token_indices = torch.where(target_token_map_bool)[0]
            target_parent_indices = self._tree.parents[: self._tree.end][
                target_token_map_bool
            ]

            input_token_map_bool = target_token_map_bool.clone()
            input_token_map_bool[target_parent_indices] = True

            input_ids = self._tree.tokens[: self._tree.end][
                input_token_map_bool
            ].unsqueeze(0)
            position_ids = self._tree.positions[: self._tree.end][
                input_token_map_bool
            ].unsqueeze(0)
            cache_seq_indices = torch.where(input_token_map_bool)[0]
            attention_mask = self._tree.amask[..., cache_seq_indices, :]

        with util.Timing(device=self._device, mode=self._target_time_mode) as wait_t:
            prefix = self._prompt if prefill else None
            target_result = asyncio.create_task(
                self._validator.request(
                    client_idx=self._client_idx,
                    req_idx=req_idx,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    cache_seq_indices=cache_seq_indices,
                    attention_mask=attention_mask,
                    parent_indices=target_parent_indices,
                    prefill=prefill,
                    prefix=prefix,
                )
            )
            await asyncio.sleep(0.00001)

            if self._proactive_client is not None:
                (
                    root_leaf_idx,
                    root_token_id,
                    proactive_tree_prefix_len,
                    proactive_tree_end,
                ) = self._proactive_client.draft()

            selection, prefill_cnt = (
                target_result.result() if target_result.done() else await target_result
            )

        with util.Timing(
            device=self._device, mode=self._target_time_mode
        ) as postprocess_t:
            interim_t = torch.ones_like(self._tree.tokens[: self._tree.end])
            interim_t[input_token_map_bool] = selection

            draft_token_choices = self._tree.tokens[: self._tree.end][
                target_token_map_bool
            ]
            target_token_choices = interim_t[target_parent_indices]

            accept_flags = draft_token_choices == target_token_choices

            accept_indices = target_token_indices[accept_flags]

            accept_mask = torch.zeros(self._tree.end, device=self._device)
            accept_mask[: self._tree.prefix_len] = 1
            accept_mask[accept_indices] = 1
            accepted_amask = attention_mask[0, 0, :, : self._tree.end] * accept_mask

            mask_row_sums = (
                attention_mask[0, 0, :, : self._tree.end].sum(dim=1).to(torch.long)
            )

            seq_lengths = accepted_amask.sum(dim=1).to(torch.long)
            best_seq_idx = (mask_row_sums * (mask_row_sums == seq_lengths)).argmax()
            best_seq_mask = attention_mask[0, 0, best_seq_idx, : self._tree.end].to(
                torch.bool
            )

            fresh_token_indices = (
                torch.where(best_seq_mask[self._tree.prefix_len :])[0]
                + self._tree.prefix_len
            )
            fresh_token_ids = self._tree.tokens[fresh_token_indices]

            last_accepted_token_idx = (
                fresh_token_indices[-1]
                if fresh_token_indices.numel() > 0
                else torch.tensor([self._tree.prefix_len - 1])
            ).to(self._device)

            # add one bonus token to num of accepted tokens
            self._logger.debug(
                "Num of accepted tokens: %d", fresh_token_indices.numel() + 1
            )

            extra_token_id = torch.tensor(
                [interim_t[last_accepted_token_idx]], device=self._device
            )

            if self._proactive_client is not None:
                self._previous_proactive_draft = self._proactive_draft

            if (
                self._proactive_client is not None
                and root_leaf_idx is not None  # type: ignore
                and root_leaf_idx == last_accepted_token_idx  # type: ignore
                and extra_token_id == root_token_id  # type: ignore
            ):
                self._proactive_draft = True
                self._reorder_by_sequence_proactive(
                    best_seq_mask,
                    proactive_tree_prefix_len,  # type: ignore
                    proactive_tree_end,  # type: ignore
                )
            else:
                self._proactive_draft = False
                self._reorder_by_sequence(best_seq_mask)
                self._tree.add(
                    token_ids=extra_token_id,
                    token_positions=self._tree.positions[self._tree.end - 1] + 1,
                    parent_indices=torch.tensor(
                        [self._tree.end - 1], device=self._device
                    ),
                    logprobs=torch.tensor([0.0], device=self._device),
                )
                self._tree.prefix_len = self._tree.end
                self._tree.status[: self._tree.prefix_len - 1] = self._tree.PROMPT

            fresh_token_ids = torch.cat(
                [fresh_token_ids, extra_token_id], dim=-1
            ).unsqueeze(0)

        stats = {
            "preprocess_t": preprocess_t.elapsed,
            "wait_t": wait_t.elapsed,
            "postprocess_t": postprocess_t.elapsed,
            "num_accepted_tokens": fresh_token_ids.size(-1),
            "prefill": prefill_cnt,
            "previous_proactive": self._previous_proactive_draft
            if self._proactive_client
            else False,
            "proactive": self._proactive_draft if self._proactive_client else False,
        }

        return fresh_token_ids, stats

    def _reorder_by_sequence(self, seq_mask: torch.Tensor):
        """
        Reorder the tree and engine's kv cache according to the validated sequence.

        Args:
            seq_mask: Sequence Mask
        """

        seq_indices = torch.where(seq_mask != 0)[0]

        self._engine.gather(
            seq_indices,
            torch.arange(seq_indices.size(-1), device=self._device),
        )

        self._tree.reorder_by_sequence(seq_mask, seq_indices)

    def _reorder_by_sequence_proactive(
        self,
        seq_mask: torch.Tensor,
        proactive_tree_prefix_len: int,
        proactive_tree_end: int,
    ):
        """
        Reorders the tree and engine's kv cache in a valid sequence
        when the tree generated by Proactive Draft is valid.

        Args:
            seq_mask: Sequence Mask
            proactive_tree_prefix_len: Start of the tree generated by Proactive Draft
            proactive_tree_end: End of the tree generated by Proactive Draft
        """
        seq_indices = torch.where(seq_mask != 0)[0]
        max_src_idx = proactive_tree_end
        mapping_tensor = torch.full(
            (max_src_idx,), -1, dtype=torch.long, device=self._device
        )

        # Original Draft Tree

        new_prefix_len = int(torch.sum(seq_mask).item())
        if torch.any(seq_mask[self._tree.prefix_len :]):
            src_indices = seq_indices[seq_indices >= self._tree.prefix_len]
            dest_indices = torch.arange(
                self._tree.prefix_len, new_prefix_len, device=self._device
            )
            mapping_tensor[src_indices] = dest_indices

            self._tree.tokens[dest_indices] = self._tree.tokens[src_indices]
            self._tree.positions[dest_indices] = dest_indices
            self._tree.parents[dest_indices] = dest_indices - 1
            self._tree.status[dest_indices] = self._tree.GENERATED

        # Proactive Tree

        src_indices = torch.arange(
            proactive_tree_prefix_len, proactive_tree_end, device=self._device
        )
        dest_indices = torch.arange(
            new_prefix_len,
            new_prefix_len + proactive_tree_end - proactive_tree_prefix_len,
            device=self._device,
        )
        mapping_tensor[src_indices] = dest_indices

        self._tree.tokens[dest_indices] = self._tree.tokens[src_indices]
        self._tree.positions[dest_indices] = self._tree.positions[src_indices]
        self._tree.parents[dest_indices] = mapping_tensor[
            self._tree.parents[src_indices]
        ]
        self._tree.status[dest_indices] = self._tree.status[src_indices]
        self._tree.logprobs[dest_indices] = self._tree.logprobs[src_indices]
        self._tree.amask[
            ...,
            dest_indices,
            new_prefix_len : new_prefix_len
            + proactive_tree_end
            - proactive_tree_prefix_len,
        ] = self._tree.amask[
            ..., src_indices, proactive_tree_prefix_len:proactive_tree_end
        ]

        self._tree.end = new_prefix_len + proactive_tree_end - proactive_tree_prefix_len
        self._tree.prefix_len = new_prefix_len + 1

        self._tree.status[: self._tree.prefix_len - 1] = self._tree.PROMPT
        self._tree.status[self._tree.prefix_len - 1 : self._tree.prefix_len + 1] = (
            self._tree.PROCESSED
        )
        self._tree.status[self._tree.status == self._tree.POST_CANDIDATE] = (
            self._tree.CANDIDATE
        )
        self._tree.status[self._tree.status == self._tree.POST_PROCESSED] = (
            self._tree.PROCESSED
        )

        self._tree.logprobs[self._tree.end :].zero_()
        # FIXME: change to public property access
        self._tree._data[:, self._tree.end :].zero_()

        _causal_mask = torch.tril(
            torch.ones(
                self._tree.prefix_len,
                self._tree.prefix_len,
                dtype=self._dtype,
                device=self._device,
            )
        )
        self._tree.amask[..., : self._tree.prefix_len, : self._tree.prefix_len] = (
            _causal_mask
        )
        self._tree.amask[
            ..., self._tree.prefix_len : self._tree.end, : self._tree.prefix_len
        ] = 1.0

        src_indices = seq_mask[: self._tree.prefix_len]
        src_indices = torch.where(src_indices)[0]
        dst_indices = torch.arange(src_indices.size(-1), device=self._device)

        self._engine.gather(src_indices, dst_indices)
