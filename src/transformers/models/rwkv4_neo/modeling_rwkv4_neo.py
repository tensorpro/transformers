# coding=utf-8
# Copyright 2022 The HuggingFace Team The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch RWKV4-Neo model. """


import math
import os

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from typing import Optional, Tuple, Union

from ...activations import ACT2FN
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import (
    apply_chunking_to_forward,
)
from ...utils import logging
from .configuration_rwkv4_neo import Rwkv4NeoConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "rwkv4-neo"
_CONFIG_FOR_DOC = "Rwkv4NeoConfig"

RWKV4_NEO_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "rwkv4-neo",
    # See all RWKV4-Neo models at https://huggingface.co/models?filter=rwkv4_neo
]


class Rwkv4NeoChannelMix(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.key = nn.Linear(config.hidden_size,
                             config.intermediate_size, bias=False)
        self.value = nn.Linear(config.intermediate_size,
                               config.hidden_size, bias=False)
        self.receptance = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False)
        self.hidden_size = config.hidden_size
        self.mix_k = torch.nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.mix_r = torch.nn.Parameter(torch.empty(1, 1, config.hidden_size))
        if isinstance(config.hidden_act, str):
            self.key_act_fn = ACT2FN[config.hidden_act]
        else:
            self.key_act_fn = config.hidden_act

    def forward(self, hidden_state, previous_input=None):
        residual = hidden_state
        hidden_state = self.layer_norm(hidden_state)
        batch_size, _, _ = hidden_state.shape
        if previous_input is None:
            previous_input = torch.zeros(batch_size, self.hidden_size)
        sx = torch.cat((previous_input.unsqueeze(1), hidden_state[:, :-1, :]), dim=1)
        print('\ncm')
        print(f'{sx.shape=}')
        print(f'{hidden_state.shape=}')
        xk = torch.lerp(sx, hidden_state, self.mix_k)
        xr = torch.lerp(sx, hidden_state, self.mix_r)
        k = self.key_act_fn(self.key(xk))
        r = F.sigmoid(self.receptance(xr))
        out = residual + r * self.value(k)
        self.out = {
                'sx': sx,
                'xk': xk,
                'xr': xr,
                'r': r,
                'k': k,
                'out': r * self.value(k),
                'x+out': out,
                'xx': hidden_state,
                'xx[-1,:]': hidden_state[-1,:],
            }
        return out, hidden_state[:, -1, :]



class Rwkv4NeoTimeMix(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.key = nn.Linear(config.hidden_size,
                             config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size,
                               config.hidden_size, bias=False)
        self.receptance = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False)
        self.output = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False)
        self.mix_r = torch.nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.mix_k = torch.nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.mix_v = torch.nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.wkv = WKV(config.hidden_size)

    def forward(self, hidden_state, previous_state=None):
        residual = hidden_state
        hidden_state = self.layer_norm(hidden_state)
        batch_size, sequence_length, _ = hidden_state.shape
        if previous_state is None:
            previous_input = torch.zeros(batch_size, self.hidden_size)
            print('from init')
        else:
            previous_input = previous_state[0]
            print('from hist')
        sx = torch.cat((previous_input.unsqueeze(1), hidden_state[:, :-1, :]), dim=1)
        print('\ntm')
        print(f'{sx.shape=}')
        print(f'{hidden_state.shape=}')
        if (previous_input.shape!= hidden_state.shape):
            print(sx.shape, hidden_state.shape, "p, h.shape")
        xk = torch.lerp(sx, hidden_state, self.mix_k)
        xv = torch.lerp(sx, hidden_state, self.mix_v)
        xr = torch.lerp(sx, hidden_state, self.mix_r)
        r = F.sigmoid(self.receptance(xr))
        k = self.key(xk)
        v = self.value(xv)
        wkv, numerator, denominator, norm = self.wkv(k, v, previous_state)
        rwkv = self.output(r * wkv)
        out = residual + rwkv
        self.out = {
            'input': residual,
            'xx': hidden_state,
            'k': k,
            'v': k,
            'kx': xk,
            'vx': xv,
            'rx': xr,
            'r': r,
            'k': k,
            'v': v,
            'aa': numerator,
            'bb': denominator,
            'pp': norm,
            'out': rwkv,
            'x+out': out,
        }
        return out, (hidden_state[:, -1, :], numerator, denominator, norm)

class WKV(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.time_decay = nn.Parameter(torch.empty(hidden_size))
        self.current_token_weight = nn.Parameter(
            torch.empty(hidden_size))

    def extra_repr(self) -> str:
        return f'hidden_size={self.hidden_size}'

    def forward(self, k, v, previous_state):
        if previous_state is None:
            numerator = denominator = torch.zeros(self.hidden_size)
            norm = torch.Tensor(self.hidden_size).fill_(-torch.inf)
        else:
            _, numerator, denominator, norm = previous_state
        sequence_length = k.shape[1]
        outputs = []
        decay = -torch.exp(self.time_decay)
        for t in range(sequence_length):
            kt = k[:, t, :]
            vt = v[:, t, :]

            current_term = self.current_token_weight + kt
            current_term_norm = torch.maximum(current_term, norm)
            previous_terms_norm = torch.exp(norm - current_term_norm)
            normed_current_term = torch.exp(current_term - current_term_norm)
            output_numerator = (previous_terms_norm * numerator + normed_current_term * vt)
            output_denominator = (previous_terms_norm * denominator + normed_current_term) 
            outputs.append(output_numerator / output_denominator)

            decay_plus_old_norm = decay + norm
            norm = torch.maximum(decay_plus_old_norm, kt)
            normed_decay = torch.exp(decay_plus_old_norm - norm)
            normed_exp_k = torch.exp(kt - norm)
            numerator = normed_decay * numerator + normed_exp_k * vt
            denominator = normed_decay * denominator + normed_exp_k

        out = torch.stack(outputs, 1)
        return out, numerator, denominator, norm


class Rwkv4NeoBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.time_mix = Rwkv4NeoTimeMix(config)
        self.channel_mix = Rwkv4NeoChannelMix(config)

    def forward(self, hidden_states, previous_state=None) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        if previous_state is not None:
            time_mix_state = previous_state[:-1]
            channel_mix_state = previous_state[-1]
        else:
            time_mix_state = channel_mix_state = None
        hidden_states, time_mix_state = self.time_mix(hidden_states, time_mix_state)
        hidden_states, channel_mix_state = self.channel_mix(hidden_states, channel_mix_state)
        return hidden_states, time_mix_state + (channel_mix_state,)


class Rwkv4NeoPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = Rwkv4NeoConfig
    base_model_prefix = "rwkv4_neo"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        pass
        # if isinstance(module, Rwkv4NeoEncoder):
        #     module.gradient_checkpointing = value


RWKV4_NEO_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config ([`~Rwkv4NeoConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

RWKV4_NEO_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`Rwkv4NeoTokenizer`].
            See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range `[0, config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert *input_ids* indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare RWKV4-Neo Model transformer outputting raw hidden-states without any specific head on top.",
    RWKV4_NEO_START_DOCSTRING,
)
class Rwkv4NeoModel(Rwkv4NeoPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    `is_decoder` argument of the configuration set to `True`.
    To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder`
    argument and `add_cross_attention` set to `True`; an
    `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.input_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.blocks = nn.ModuleList([Rwkv4NeoBlock(config) for _ in range(config.num_hidden_layers)])
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        # return self.foo
        return None
        return self.embeddings

    def set_input_embeddings(self, value):
        return
        self.embeddings = value

    @add_start_docstrings_to_model_forward(RWKV4_NEO_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape `(batch_size, 1)`
            instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up
            decoding (see `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        

        # if len(inputs_embeds.shape) == 1:
        #     inputs_embeds = inputs_embeds.unsqueeze(0)
        hidden_states = self.ln_out = self.input_layer_norm(inputs_embeds)
    
        # past_key_values_length
        # past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        from itertools import repeat
        if past_key_values is None:
            past_key_values = repeat(None)

        rnn_state = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, block_past) in enumerate(zip(self.blocks, past_key_values)):
            hidden_states, block_state = block(
                hidden_states,
                block_past,
            )
            if all_hidden_states is not None:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if use_cache is True:
                rnn_state = rnn_state + (block_state,)

        hidden_states = self.output_layer_norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, rnn_state, all_hidden_states] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=rnn_state,
            hidden_states=all_hidden_states,
        )



@add_start_docstrings(
    """RWKV4-Neo Model with a `language modeling` head on top for CLM fine-tuning. """, RWKV4_NEO_START_DOCSTRING
)
class Rwkv4NeoForCausalLM(Rwkv4NeoPreTrainedModel):

    _keys_to_ignore_on_load_missing = [
        r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning(
                "If you want to use `Rwkv4NeoForCausalLM` as a standalone, add `is_decoder=True.`")

        self.model = Rwkv4NeoModel(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # self.embeddings = nn.Embedding(1,1)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    # def get_output_embeddings(self):
    #     return self.head

    # def set_output_embeddings(self, new_embeddings):
    #     self.head = new_embeddings

    @add_start_docstrings_to_model_forward(RWKV4_NEO_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2
            tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional
            tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two
            additional tensors are only required when the model is used as a decoder in a Sequence to Sequence
            model.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
            cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential
            decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape `(batch_size, 1)`
            instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up
            decoding (see `past_key_values`).

        Returns:

        Example:

        ```python
        >>> from transformers import Rwkv4NeoTokenizer, Rwkv4NeoForCausalLM, Rwkv4NeoConfig
        >>> import torch

        >>> tokenizer = Rwkv4NeoTokenizer.from_pretrained('rwkv4-neo')
        >>> config = Rwkv4NeoConfig.from_pretrained("rwkv4-neo")
        >>> config.is_decoder = True
        >>> model = Rwkv4NeoForCausalLM.from_pretrained('rwkv4-neo', config=config)

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.logits
        ```
"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.head(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:,
                                                          :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(
                shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx)
                               for past_state in layer_past[:2]) + layer_past[2:],)
        return reordered_past
