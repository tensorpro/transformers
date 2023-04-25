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


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


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


@torch.no_grad()
def convert_rwkv_checkpoint(checkpoint_path, pytorch_dump_folder_path,):
    """
    Copy/paste/tweak model's weights to our RWKV structure.
    """
    state_dict = torch.load(checkpoint_path, 'cpu')

    remove_ignore_keys_(state_dict)
    keys = list(state_dict.keys())
    for k in keys:
        rename_key(state_dict, k)

    num_layers = max(int(key.split('.')[2])
                     for key in state_dict if 'blocks.' in key) + 1
    vocab_size, hidden_size = state_dict['model.embeddings.weight'].shape
    cfg = Rwkv4NeoConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    model = Rwkv4NeoForCausalLM(cfg)
    model.load_state_dict(state_dict)
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--hf_repo", type=str, help="path in a file system to an RWKV model",
    )
    parser.add_argument(
        "--hf_model_path", type=str, help="specific model path in hugginface repo",
    )
    parser.add_argument(
        "--local_path", type=str, help="path in a file system to an RWKV model",
    )
    parser.add_argument("--output_path", default=None,
                        type=str, help="Path to the output PyTorch model.",
                        )
    args = parser.parse_args()

    valid_hf = args.hf_repo and args.hf_model_path
    valid_path = args.local_path

    if not (valid_hf or valid_path):
        raise ValueError('You must specify either --local_path, or both [--hf_repo_path and --hf_model_path]')
    
    if valid_hf and valid_path:
        raise ValueError('You cannot set both --local_path, and [--hf_repo_path and --hf_model_path].'
                         'Use --local_path for a checkpoint on your filesystem, and use --hf_repo/--hf_model_path'
                         'to load a hub model')
    path = args.local_path
    print('ayy')
    if valid_hf:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(args.hf_repo, args.hf_model_path)
        print(path)
    convert_rwkv_checkpoint(os.path.expanduser(path), os.path.expanduser(args.output_path))
