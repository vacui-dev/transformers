# Copyright 2025 Vacui.dev and The HuggingFace Inc. team. All rights reserved.

from typing import Callable, Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import logging
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaMLP,
    LlamaPreTrainedModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..mistral.modeling_mistral import MistralModel
from .configuration_vacui03 import Vacui03Config


logger = logging.get_logger(__name__)


class Vacui03MLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)


class Vacui03Attention(LlamaAttention):
    def __init__(self, config: Vacui03Config, layer_idx: int):
        super().__init__()
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear((config.num_attention_heads * self.head_dim) + config.c_subspace_dim, config.hidden_size, bias=True)


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        
        # Split hidden states into Q/K/V subspaces and C subspace
        qkv_subspace = hidden_states[..., :self.config.qkv_subspace_dim]
        c_subspace = hidden_states[..., self.config.qkv_subspace_dim:]
        
        # Prepare Q/K/V inputs, by adding one-directional C influence
        # (C is detached to prevent backpropagation through it)
        q_input = torch.cat([qkv_subspace, c_subspace.detach()], dim=-1)
        k_input = torch.cat([qkv_subspace, c_subspace.detach()], dim=-1)
        v_input = torch.cat([qkv_subspace, c_subspace.detach()], dim=-1)
        
        # Project Q/K
        query_states = self.q_proj(q_input)
        key_states = self.k_proj(k_input)
        value_states = self.v_proj(v_subspace)
            
        # Reshape for attention computation
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        # c subspace is NOT detached here -- allow training of context based on the output
        o_input = torch.cat([attn_output, c_subspace], dim=-1)
        attn_output = self.o_proj(o_input)
        
        return attn_output, attn_weights


class Vacui03DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: Vacui03Config, layer_idx: int):
        super().__init__()
        self.self_attn = Vacui03Attention(config=config, layer_idx=layer_idx)
        self.mlp = Vacui03MLP(config)


class Vacui03PreTrainedModel(LlamaPreTrainedModel):
    pass


class Vacui03Model(MistralModel):
    pass


class Vacui03ForCausalLM(LlamaForCausalLM):
    pass


class Vacui03ForSequenceClassification(LlamaForSequenceClassification):
    pass


class Vacui03ForTokenClassification(LlamaForTokenClassification):
    pass


class Vacui03ForQuestionAnswering(LlamaForQuestionAnswering):
    pass


__all__ = [
    "Vacui03PreTrainedModel",
    "Vacui03Model",
    "Vacui03ForCausalLM",
    "Vacui03ForSequenceClassification",
    "Vacui03ForTokenClassification",
    "Vacui03ForQuestionAnswering",
]
