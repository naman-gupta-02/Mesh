"""A contiguous slice of a GPT-2 model's transformer blocks.

This is the unit of work a node holds: the embedding + first N blocks on the
entry shard, some middle blocks on intermediate shards, and the final
layernorm + LM head on the exit shard. Chaining shard.forward() calls in
order reproduces the monolithic model's forward pass exactly.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel


class ModelShard(nn.Module):
    def __init__(
        self,
        model: GPT2LMHeadModel,
        layer_start: int,
        layer_end: int,
        include_embed: bool,
        include_head: bool,
    ):
        super().__init__()
        transformer = model.transformer
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.include_embed = include_embed
        self.include_head = include_head

        if include_embed:
            self.wte = transformer.wte
            self.wpe = transformer.wpe
            self.drop = transformer.drop

        self.blocks = nn.ModuleList(transformer.h[layer_start:layer_end])

        if include_head:
            self.ln_f = transformer.ln_f
            self.lm_head = model.lm_head

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.include_embed:
            position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long).unsqueeze(0)
            hidden_states = self.wte(input_ids) + self.wpe(position_ids)
            hidden_states = self.drop(hidden_states)

        for block in self.blocks:
            hidden_states = block(hidden_states)[0]

        if self.include_head:
            hidden_states = self.ln_f(hidden_states)
            hidden_states = self.lm_head(hidden_states)

        return hidden_states
