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
from .configuration_vacui02 import Vacui02Config


logger = logging.get_logger(__name__)


class Vacui02MLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)


class Vacui02Attention(LlamaAttention):
    def __init__(self, config: Vacui02Config, layer_idx: int):
        super().__init__()
        self.q_proj = nn.Linear(config.qk_subspace_dim + config.a_subspace_dim + config.v_context_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.qk_subspace_dim + config.a_subspace_dim + config.v_context_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.a_proj = nn.Linear(config.a_subspace_dim + config.v_context_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.v_subspace_dim, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(
            (config.num_attention_heads * self.head_dim) + config.v_subspace_dim, config.hidden_size, bias=config.attention_bias
        )
        
        self.v_to_context = nn.Linear(config.v_subspace_dim, config.v_context_size, bias=True)


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
        
        # Split hidden states into Query/Key, Value and Action subspaces
        qka_dim = config.qk_subspace_dim + config.a_subspace_dim
        qka_subspace = hidden_states[..., :config.qk_subspace_dim + config.a_subspace_dim]
        a_subspace = qka_subspace[..., :config.a_subspace_dim]
        v_subspace = hidden_states[..., qka_dim:]

        # detach v_subspace() to make the v_context for qka read-only.
        # qka can train off of v, but v is not influenced by qka.
        # similarly, qk and a do not influence each other directly
        v_context_for_qka = self.v_to_context(v_subspace.detach())
        
        # Prepare Q/K inputs, by adding one-directional V influence
        q_input = torch.cat([qka_subspace, v_context_for_qka], dim=-1)
        k_input = torch.cat([qka_subspace, v_context_for_qka], dim=-1)
        a_input = torch.cat([a_subspace, v_context_for_qka], dim=-1)
        
        # Project Q/K/V/A
        query_states = self.q_proj(q_input)
        key_states = self.k_proj(k_input)
        action_states = self.a_proj(a_input)
        value_states = self.v_proj(v_subspace)
            
        # Reshape for attention computation
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)
        action_states = action_states.view(hidden_shape).transpose(1, 2)

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
        o_input = torch.cat([attn_output, v_subspace], dim=-1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights


class Vacui02DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: Vacui02Config, layer_idx: int):
        super().__init__()
        self.self_attn = Vacui02Attention(config=config, layer_idx=layer_idx)
        self.mlp = Vacui02MLP(config)


class Vacui02PreTrainedModel(LlamaPreTrainedModel):
    pass


class Vacui02Model(MistralModel):
    pass


class Vacui02ForCausalLM(LlamaForCausalLM):
    pass


class Vacui02ForSequenceClassification(LlamaForSequenceClassification):
    pass


class Vacui02ForTokenClassification(LlamaForTokenClassification):
    pass


class Vacui02ForQuestionAnswering(LlamaForQuestionAnswering):
    pass


__all__ = [
    "Vacui02PreTrainedModel",
    "Vacui02Model",
    "Vacui02ForCausalLM",
    "Vacui02ForSequenceClassification",
    "Vacui02ForTokenClassification",
    "Vacui02ForQuestionAnswering",
]
