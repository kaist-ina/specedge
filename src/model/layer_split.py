import torch

from model.cache import KVCache


class PartialFirstHalf(torch.nn.Module):
    def __init__(self, full_model, num_layers: int):
        super().__init__()
        self.config = full_model.config
        self.embed_tokens = full_model.model.embed_tokens
        self.rotary_emb = full_model.model.rotary_emb
        self.layers = torch.nn.ModuleList()

        self.device = torch.device("cuda:0")
        self.dtype = torch.float16

        for idx in range(num_layers):
            layer = full_model.model.layers[idx]
            self.layers.append(layer)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_batch_indices: torch.Tensor,
        cache_seq_indices: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: KVCache,
    ):
        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
            )

            hidden_states = layer_outputs[0]

        return hidden_states, position_embeddings, past_key_values


class PartialSecondHalf(torch.nn.Module):
    def __init__(self, full_model, num_layers: int, offset: int):
        super().__init__()
        self.config = full_model.config
        self.layers = torch.nn.ModuleList()
        self.rotary_emb = full_model.model.rotary_emb
        self.norm = full_model.model.norm
        self.lm_head = full_model.lm_head

        for idx in range(offset, num_layers):
            layer = full_model.model.layers[idx]
            self.layers.append(layer)

        self.device = torch.device("cuda:0")
        self.dtype = torch.float16

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_batch_indices: torch.Tensor,
        cache_seq_indices: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: KVCache,
    ):
        position_embeddings = self.rotary_emb(input_ids, position_ids)
        hidden_states = input_ids
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        lm_logits = self.lm_head(hidden_states)

        return lm_logits, past_key_values
