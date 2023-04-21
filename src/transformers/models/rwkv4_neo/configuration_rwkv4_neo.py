# coding=utf-8
# Copyright 2022 The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
""" RWKV4-Neo model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

RWKV4_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "rwkv4-neo": "https://huggingface.co/rwkv4-neo/resolve/main/config.json",
    # See all RWKV4-Neo models at https://huggingface.co/models?filter=rwkv4_neo
}


class Rwkv4NeoConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~Rwkv4NeoModel`].
    It is used to instantiate an RWKV4-Neo model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the RWKV4-Neo [rwkv4-neo](https://huggingface.co/rwkv4-neo) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the RWKV4-Neo model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~Rwkv4NeoModel`] or
            [`~TFRwkv4NeoModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 4*hidden_size):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last hidden state (not used by all models). Only
            relevant if `config.is_decoder=True`.
        Example:

    ```python
    >>> from transformers import Rwkv4NeoModel, Rwkv4NeoConfig

    >>> # Initializing a RWKV4-Neo rwkv4-neo style configuration
    >>> configuration = Rwkv4NeoConfig()

    >>> # Initializing a model from the rwkv4-neo style configuration
    >>> model = Rwkv4NeoModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
"""
    model_type = "rwkv4_neo"
    

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        intermediate_size=None,
        hidden_act="relu2",
        initializer_range=0.02,
        layer_norm_eps=1e-05,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.is_decoder=True
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.tie_encoder_decoder = False
        self.tie_word_embeddings = False
        self.num_attention_heads = 1
        self.intermediate_size = intermediate_size or 4 * hidden_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )

    