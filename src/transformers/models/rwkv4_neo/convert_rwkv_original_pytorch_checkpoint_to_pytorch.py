# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""Convert BART checkpoint."""


import argparse
import os
from pathlib import Path

import torch
from packaging import version
from torch import nn

from transformers import (
    Rwkv4NeoForCausalLM,
    Rwkv4NeoConfig,
)

from transformers.utils import logging


FAIRSEQ_MODELS = ["bart.large", "bart.large.mnli",
                  "bart.large.cnn", "bart_xsum/model.pt"]

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

SAMPLE_TEXT = " Hello world! cécé herlolip"

rename_keys_dict = {
    'emb.weight': 'model.embeddings.weight',
    'blocks.0.ln0.weight': 'model.input_layer_norm.weight',
    'blocks.0.ln0.bias': 'model.input_layer_norm.bias',
    'ln_out.weight': 'model.output_layer_norm.weight',
    'ln_out.bias': 'model.output_layer_norm.bias',
    'head.weight': 'head.weight',
}

key_substring_replacements = [
    ('.time_mix_', '.mix_'),
    ('.time_decay', '.wkv.time_decay'),
    ('.time_first', '.wkv.current_token_weight'),
    ('.att', '.time_mix'),
    ('.ffn', '.channel_mix'),
    ('.ln1', '.time_mix.layer_norm'),
    ('.ln2', '.channel_mix.layer_norm'),
]

def get_new_key(old_key: str) -> str:

    if old_key in rename_keys_dict:
        return rename_keys_dict[old_key]
    new_key = f'model.{old_key}'
    for old_substring, new_substring in key_substring_replacements:
        new_key = new_key.replace(old_substring, new_substring)
    return new_key


def rename_key(dct, old):
    val = dct.pop(old)
    new = get_new_key(old)
    dct[new] = val


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)

def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


@torch.no_grad()
def convert_rwkv_checkpoint(checkpoint_path, pytorch_dump_folder_path, repo, hf_checkpoint_name=None):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    # if not os.path.exists(checkpoint_path):
    #     rwkv = torch.hub.load("BlinkDL/fairseq", checkpoint_path).eval()
    # else:
    state_dict = torch.load(checkpoint_path, 'cpu')

    # rwkv.model.upgrade_state_dict(rwkv.model.state_dict())
    if hf_checkpoint_name is None:
        hf_checkpoint_name = checkpoint_path.replace(".", "-")
    # config = Rwkv4NeoConfig.from_pretrained(hf_checkpoint_name)
    # tokens = bart.encode(SAMPLE_TEXT).unsqueeze(0)
    # tokens2 = BartTokenizer.from_pretrained(hf_checkpoint_name).encode(
    #     SAMPLE_TEXT, return_tensors="pt").unsqueeze(0)
    # assert torch.eq(tokens, tokens2).all()

    # state_dict = rwkv.state_dict()
    remove_ignore_keys_(state_dict)
    keys = list(state_dict.keys())
    for k in keys:

        if 'emb' in k:
            # state_dict[k] = state_dict[k]
            print(state_dict[k])


        # if 'time_mix_' in k:
            # print(k)
            # state_dict[k] = state_dict[k].squeeze()
            # print(state_dict[k].shape)
        rename_key(state_dict, k)

    return state_dict
    # if checkpoint_path == "bart.large.mnli":
    #     state_dict = rwkv.state_dict()
    #     remove_ignore_keys_(state_dict)
    #     state_dict["model.shared.weight"] = state_dict["model.decoder.embed_tokens.weight"]
    #     for src, dest in mnli_rename_keys:
    #         rename_key(state_dict, src, dest)
    #     model = BartForSequenceClassification(config).eval()
    #     model.load_state_dict(state_dict)
    #     fairseq_output = bart.predict("mnli", tokens, return_logits=True)
    #     new_model_outputs = model(tokens)[0]  # logits
    # else:  # no classification heads to worry about
    #     state_dict = bart.model.state_dict()
    #     remove_ignore_keys_(state_dict)
    #     state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
    #     fairseq_output = bart.extract_features(tokens)
    #     if hf_checkpoint_name == "facebook/bart-large":
    #         model = BartModel(config).eval()
    #         model.load_state_dict(state_dict)
    #         new_model_outputs = model(tokens).model[0]
    #     else:
    #         model = BartForConditionalGeneration(
    #             config).eval()  # an existing summarization ckpt
    #         model.model.load_state_dict(state_dict)
    #         if hasattr(model, "lm_head"):
    #             model.lm_head = make_linear_from_emb(model.model.shared)
    #         new_model_outputs = model.model(tokens)[0]

    # Check results
    # assert fairseq_output.shape == new_model_outputs.shape
    # assert (fairseq_output == new_model_outputs).all().item()
    # Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--repo", type=str, help="hugging face repo", default=None,
    )
    parser.add_argument(
        "path", type=str, help="path in a huggingface space",
    )
    parser.add_argument("--pytorch_dump_folder_path", default=None,
                        type=str, help="Path to the output PyTorch model.",
                        )
    parser.add_argument(
        "--hf_config", default=None, type=str, help="Which huggingface architecture to use: bart-large-xsum",
    )
    args = parser.parse_args()
    a = convert_rwkv_checkpoint(
        args.path, args.pytorch_dump_folder_path, args.repo, hf_checkpoint_name=args.hf_config)
    print(list(a.keys()))
    cfg = Rwkv4NeoConfig(vocab_size=50277)
    m = Rwkv4NeoForCausalLM(cfg)
    print(m.model.embeddings.state_dict())
    m.load_state_dict(a)
    print(m.model.embeddings.state_dict())
    print(a['model.embeddings.weight'])
    print(m(torch.arange(10).unsqueeze(0)))
