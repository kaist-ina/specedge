from typing import Optional

import torch

import log
from config import SpecEdgeBatchClientConfig as config
from specedge.engine.graph import BatchGraphEngine
from specedge.tree import BatchTree
from strategy.request_manager import RequestManager


class SpecExecEdgeVerify:
    def __init__(
        self,
        tree: BatchTree,
        eos_token: torch.Tensor,
        req_manager: RequestManager,
        draft_engine: Optional[BatchGraphEngine] = None,
        target_engine: Optional[BatchGraphEngine] = None,
    ):
        self._logger = log.get_logger()
        self._result_logger = log.get_result_logger()

        self._req_manager = req_manager
        self._eos_token = eos_token

        # Load configuration
        self._device = config.device
        self._dtype = config.dtype

        self._max_len = config.max_len
        self._max_n_beams = config.max_n_beams
        self._max_beam_len = config.max_beam_len
        self._max_branch_width = config.max_branch_width
        self._max_batch_size = config.max_batch_size
        self._max_budget = config.max_budget
        self._max_new_tokens = config.max_new_tokens

        self._tree = tree

        self._draft_engine = draft_engine
        self._target_engine = target_engine

        # predefine indices
        self._col_indices = torch.arange(self._max_len, device=self._device)
        self._batch_indices = torch.arange(
            self._max_batch_size, device=self._device
        ).repeat_interleave(self._max_budget + 1)

    async def edge_pre_verify(self):
        # shape: [batch_size, max_len]
        target_token_map_bool = self._tree.status >= self._tree.PROCESSED

        # prefix mask
        prefix_mask = self._col_indices < (self._tree.prefix_len.unsqueeze(1) - 1)
        target_token_map_bool[prefix_mask] = False

        end_mask = self._col_indices > (self._tree.end.unsqueeze(1) - 1)
        target_token_map_bool[end_mask] = False

        indices = target_token_map_bool.nonzero(as_tuple=False)
        batch_indices = indices[:, 0]
        seq_indices = indices[:, 1]

        offset = self._tree.prefix_len[batch_indices] - 1
        adjusted_seq_indices = seq_indices - offset

        # get input_ids
        input_ids = torch.zeros(
            (self._max_batch_size, self._max_budget + 1),
            dtype=torch.long,
            device=self._device,
        )
        input_ids[batch_indices, adjusted_seq_indices] = self._tree.tokens[
            batch_indices, seq_indices
        ]

        # get position_ids
        position_ids = torch.full(
            (self._max_batch_size, self._max_budget + 1),
            self._max_len - 1,
            device=self._device,
        )
        position_ids[batch_indices, adjusted_seq_indices] = self._tree.positions[
            batch_indices, seq_indices
        ]

        # get cache_batch_indices, cache_seq_indices
        cache_batch_indices = self._batch_indices
        cache_seq_indices = torch.full(
            (self._max_batch_size, self._max_budget + 1),
            self._max_len - 1,
            device=self._device,
        )
        cache_seq_indices[batch_indices, adjusted_seq_indices] = seq_indices
        cache_seq_indices = cache_seq_indices.flatten()

        # get attention mask
        attention_mask = torch.zeros(
            (self._max_batch_size, 1, self._max_budget + 1, self._max_len),
            dtype=self._tree.amask.dtype,
            device=self._device,
        )
        attention_mask[batch_indices, :, adjusted_seq_indices, :] = self._tree.amask[
            batch_indices, :, seq_indices, :
        ]

        return (
            input_ids,
            position_ids,
            batch_indices,
            cache_batch_indices,
            cache_seq_indices,
            attention_mask,
            adjusted_seq_indices,
            seq_indices,
        )

    async def edge_post_verify(
        self,
        selection: torch.Tensor,
        batch_indices: torch.Tensor,
        adjusted_seq_indices: torch.Tensor,
        original_seq_indices: torch.Tensor,
    ):
        selection = selection.to(self._device)
        draft_choices = torch.ones(
            (self._max_batch_size, self._max_budget + 1),
            dtype=torch.long,
            device=self._device,
        )

        tree_batch_indices = torch.zeros_like(
            draft_choices, dtype=torch.long, device=self._device
        )

        tree_seq_indices = torch.full_like(
            draft_choices, self._max_len - 1, dtype=torch.long, device=self._device
        )

        draft_choices[batch_indices, adjusted_seq_indices] = self._tree.tokens[
            batch_indices, original_seq_indices
        ]

        tree_batch_indices[batch_indices, adjusted_seq_indices] = batch_indices
        tree_seq_indices[batch_indices, adjusted_seq_indices] = original_seq_indices

        draft_choices = draft_choices[:, 1:]
        tree_batch_indices = tree_batch_indices[:, 1:]
        tree_seq_indices = tree_seq_indices[:, 1:]

        target_choices = torch.zeros(
            (self._max_batch_size, self._max_budget + 1),
            dtype=torch.long,
            device=self._device,
        )

        offset = self._tree.prefix_len[batch_indices] - 1

        target_choices[batch_indices, adjusted_seq_indices] = selection[
            batch_indices,
            (self._tree.parents[batch_indices, original_seq_indices] - offset),
        ]

        target_choices = target_choices[:, 1:]

        accepted = draft_choices == target_choices
        accepted_batch_indices = tree_batch_indices[accepted]
        accepted_seq_indices = tree_seq_indices[accepted]

        crit_mask = torch.zeros(
            (self._max_batch_size, self._max_len), dtype=torch.bool, device=self._device
        )
        crit_mask[accepted_batch_indices, accepted_seq_indices] = True
        prefix_mask = self._col_indices < self._tree.prefix_len.unsqueeze(1)
        crit_mask |= prefix_mask

        accepted_mask = self._tree.amask[
            accepted_batch_indices, 0, accepted_seq_indices, :
        ].to(torch.bool)
        accepted_mask = accepted_mask.sum(dim=-1) == (
            crit_mask[accepted_batch_indices] * accepted_mask
        ).sum(dim=-1)

        accepted_batch_indices = accepted_batch_indices[accepted_mask]
        accepted_seq_indices = accepted_seq_indices[accepted_mask]
        accepted_positions = self._tree.positions[
            accepted_batch_indices, accepted_seq_indices
        ]

        bonus_token_logprobs = torch.zeros(
            (self._max_batch_size,), dtype=torch.float32, device=self._device
        )
        eos_flag = torch.zeros(
            (self._max_batch_size,), dtype=torch.bool, device=self._device
        )

        raw_batch_indices = torch.arange(
            self._max_batch_size, dtype=torch.long, device=self._device
        )

        if accepted_batch_indices.numel() > 0:
            crit = torch.max(
                torch.where(
                    raw_batch_indices.unsqueeze(1) == accepted_batch_indices,
                    accepted_positions,
                    -1,
                ),
                dim=-1,
            )

            n_fresh_tokens = torch.clamp_min(
                crit.values - (self._tree.prefix_len - 1) + 1, min=1
            )
            max_pos_indices = torch.where(
                crit.values != -1,
                accepted_seq_indices[crit.indices],
                self._tree.prefix_len + 1,
            )
        else:
            max_pos_indices = self._tree.prefix_len - 1
            n_fresh_tokens = torch.ones_like(max_pos_indices, device=self._device)

        src_masks = self._tree.amask[raw_batch_indices, 0, max_pos_indices, :].to(
            torch.bool
        )
        bonus_token_ids = selection[
            raw_batch_indices, max_pos_indices - self._tree.prefix_len + 1
        ]

        eos_flag = bonus_token_ids == self._eos_token
        eos_flag |= (
            self._tree.tokens[raw_batch_indices] & src_masks == self._eos_token
        ).any(dim=-1)

        for b_idx in range(self._max_batch_size):
            self._reorder_by_sequence(b_idx, src_masks[b_idx])

        bonus_token_positions = (
            torch.gather(
                self._tree.positions,
                dim=1,
                index=self._tree.prefix_len.unsqueeze(1) - 1,
            )
            + 1
        ).T.squeeze(0)

        self._tree.add(
            batch_indices=torch.arange(
                self._max_batch_size, dtype=torch.long, device=self._device
            ),
            token_ids=bonus_token_ids,
            token_positions=bonus_token_positions,
            parent_indices=self._tree.prefix_len - 1,
            logprobs=bonus_token_logprobs,
        )

        self._tree.prefix_len.copy_(self._tree.end)
        self._remove_request(eos_flag)

        return n_fresh_tokens

    def _reorder_by_sequence(self, batch_idx: int, src_mask: torch.Tensor):
        src_indices = torch.where(src_mask)[0]
        dest_indices = torch.arange(src_indices.numel(), device=self._device)

        if self._draft_engine:
            self._draft_engine.gather(batch_idx, src_indices, dest_indices)
        if self._target_engine:
            self._target_engine.gather(batch_idx, src_indices, dest_indices)

        self._tree.reorder(batch_idx, src_indices, dest_indices)

    def _remove_request(self, eos_flag: torch.Tensor):
        max_token_reached = (
            self._tree.prefix_len - self._req_manager.initial_prefix_len
        ) > self._max_new_tokens
        empty_req_flag = self._req_manager.initial_prefix_len != 0
        remove_flag = (max_token_reached | eos_flag) & empty_req_flag

        if remove_flag.any():
            remove_indices = torch.where(remove_flag)[0]
            self._logger.debug("Removing requests %s", remove_indices.tolist())

            self._tree.remove_requests(remove_indices)
            self._req_manager.remove_requests(remove_indices)

            if self._draft_engine:
                self._draft_engine.remove_requests(remove_indices)

            if self._target_engine:
                self._target_engine.remove_requests(remove_indices)
