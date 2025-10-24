from typing import Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from transformers.models.llama import LlamaPreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

from .cache import KVCache


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            self.config,
            "head_dim",
            self.config.hidden_size // self.config.num_attention_heads,
        )
        self.num_key_value_groups = (
            self.config.num_attention_heads // self.config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.config.hidden_size,
            self.config.num_attention_heads * self.head_dim,
            bias=self.config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.config.hidden_size,
            self.config.num_key_value_heads * self.head_dim,
            bias=self.config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.config.hidden_size,
            self.config.num_key_value_heads * self.head_dim,
            bias=self.config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.config.num_attention_heads * self.head_dim,
            self.config.hidden_size,
            bias=self.config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_key_value: KVCache,
        cache_batch_indices: torch.Tensor,
        cache_seq_indices: torch.Tensor,
    ):
        input_shape = hidden_states.size()[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        key_states, value_states = past_key_value.update(
            key_states,
            value_states,
            self.layer_idx,
            cache_batch_indices,
            cache_seq_indices,
        )

        attn_output = self._sdpa_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output

    def _sdpa_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
        dropout: float = 0.0,
    ):
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.scaling,
            is_causal=False,
        )

        return attn_output.transpose(1, 2).contiguous()


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config=config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_value: KVCache,
        cache_batch_indices: torch.Tensor,
        cache_seq_indices: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn.forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_batch_indices=cache_batch_indices,
            cache_seq_indices=cache_seq_indices,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states,)


class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        cache_batch_indices: torch.LongTensor,
        cache_seq_indices: torch.LongTensor,
        attention_mask: torch.Tensor,
        past_key_values: KVCache,
    ):
        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_values,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        return hidden_states, past_key_values


class LlamaForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config):
        self._tied_weights_keys = ["lm_head.weight"]
        self._tp_plan = {"lm_head": "colwise_rep"}
        self._pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        cache_batch_indices: torch.LongTensor,
        cache_seq_indices: torch.LongTensor,
        attention_mask: torch.Tensor,
        past_key_values: KVCache,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_batch_indices=cache_batch_indices,
            cache_seq_indices=cache_seq_indices,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        logits = self.lm_head(hidden_states)

        return logits, past_key_values
