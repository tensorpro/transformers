# Copyright 2020 The HuggingFace Team. All rights reserved.
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
from typing import TYPE_CHECKING

from ...utils import  _LazyModule, OptionalDependencyNotAvailable, is_tokenizers_available
from ...utils import is_torch_available



from ...utils import is_flax_available




_import_structure = {
    "configuration_rwkv4_neo": ["RWKV4_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP", "Rwkv4NeoConfig"],
    "tokenization_rwkv4_neo": ["Rwkv4NeoTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_rwkv4_neo_fast"] = ["Rwkv4NeoTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_rwkv4_neo"] = [
        "RWKV4_NEO_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Rwkv4NeoForCausalLM",
        "Rwkv4NeoBlock",
        "Rwkv4NeoChannelMix",
        "Rwkv4NeoTimeMix",
        "Rwkv4NeoModel",
        "Rwkv4NeoPreTrainedModel",
    ]



# try:
#     if not is_flax_available():
#         raise OptionalDependencyNotAvailable()
# except OptionalDependencyNotAvailable:
#     pass
# else:
#     _import_structure["modeling_flax_rwkv4_neo"] = [
#         "FlaxRwkv4NeoForCausalLM",
#         "FlaxRwkv4NeoBlock",
#         "FlaxRwkv4NeoModel",
#         "FlaxRwkv4NeoPreTrainedModel",
#     ]




if TYPE_CHECKING:
    from .configuration_rwkv4_neo import RWKV4_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP, Rwkv4NeoConfig
    from .tokenization_rwkv4_neo import Rwkv4NeoTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_rwkv4_neo_fast import Rwkv4NeoTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_rwkv4_neo import (
            RWKV4_NEO_PRETRAINED_MODEL_ARCHIVE_LIST,
            Rwkv4NeoForMaskedLM,
            Rwkv4NeoForCausalLM,
            Rwkv4NeoForMultipleChoice,
            Rwkv4NeoForQuestionAnswering,
            Rwkv4NeoForSequenceClassification,
            Rwkv4NeoForTokenClassification,
            Rwkv4NeoLayer,
            Rwkv4NeoModel,
            Rwkv4NeoPreTrainedModel,
            load_tf_weights_in_rwkv4_neo,
        )



    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_rwkv4_neo import (
            FlaxRwkv4NeoForMaskedLM,
            FlaxRwkv4NeoForCausalLM,
            FlaxRwkv4NeoForMultipleChoice,
            FlaxRwkv4NeoForQuestionAnswering,
            FlaxRwkv4NeoForSequenceClassification,
            FlaxRwkv4NeoForTokenClassification,
            FlaxRwkv4NeoLayer,
            FlaxRwkv4NeoModel,
            FlaxRwkv4NeoPreTrainedModel,
        )



else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
