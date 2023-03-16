# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
# Modified by Brendan Dolan-Gavitt, 2022
# Modified by Meng Zhang, 2023
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

import argparse
import configparser
import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import GPTJForCausalLM

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../..")
sys.path.append(dir_path)


def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    else:
        assert False, f"Invalid weight data type {data_type}"


def split_and_convert_process(i, saved_dir, factor, key, val):
    if (
        key.find("input_layernorm.weight") != -1
        or key.find("input_layernorm.bias") != -1
        or key.find("attention.dense.bias") != -1
        or key.find("post_attention_layernorm.weight") != -1
        or key.find("post_attention_layernorm.bias") != -1
        or key.find("mlp.dense_4h_to_h.bias") != -1
        or key.find("final_layernorm.weight") != -1
        or key.find("final_layernorm.bias") != -1
    ):

        # shared weights, only need to convert the weights of rank 0
        if i == 0:
            saved_path = saved_dir + "/model." + key + ".bin"
            val.tofile(saved_path)

    elif (
        key.find("attention.dense.weight") != -1
        or key.find("mlp.dense_4h_to_h.weight") != -1
    ):
        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            saved_path = saved_dir + "/model." + key + ".%d.bin" % (i * factor + j)
            split_vals[j].tofile(saved_path)

    elif (
        key.find("mlp.dense_h_to_4h.weight") != -1
        or key.find("mlp.dense_h_to_4h.bias") != -1
    ):

        split_vals = np.split(val, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir + "/model." + key + ".%d.bin" % (i * factor + j)
            split_vals[j].tofile(saved_path)

    elif key.find("attention.query_key_value.weight") != -1:
        split_vals = np.split(val, factor, axis=-1)

        for j in range(factor):
            saved_path = saved_dir + "/model." + key + ".%d.bin" % (i * factor + j)
            split_vals[j].tofile(saved_path)

    else:
        print("[ERROR] cannot find key '{}'".format(key))


def split_and_convert(args):
    saved_dir = args.saved_dir + "/%d-gpu/" % args.infer_gpu_num

    if os.path.exists(saved_dir) is False:
        os.makedirs(saved_dir)

    t_gpu_num = args.trained_gpu_num
    i_gpu_num = args.infer_gpu_num
    assert i_gpu_num % t_gpu_num == 0

    factor = (int)(i_gpu_num / t_gpu_num)

    model = GPTJForCausalLM.from_pretrained(args.in_file)

    try:
        config = configparser.ConfigParser()
        config["gpt"] = {}
        for key in vars(args):
            config["gpt"][key] = f"{vars(args)[key]}"
        for k, v in vars(model.config).items():
            config["gpt"][k] = f"{v}"
        config["gpt"]["weight_data_type"] = args.weight_data_type
        with open((Path(saved_dir) / "config.ini").as_posix(), "w") as configfile:
            config.write(configfile)
    except Exception:
        print("Fail to save the config in config.ini.")
    np_weight_data_type = get_weight_data_type(args.weight_data_type)

    huggingface_model_name_pattern = [
        "ln_1.bias",
        "ln_1.weight",
        "attn.q_proj.weight",
        "attn.out_proj.weight",
        "mlp.fc_in.bias",
        "mlp.fc_in.weight",
        "mlp.fc_out.bias",
        "mlp.fc_out.weight",
    ]

    ft_model_name_pattern = [
        "input_layernorm.bias",
        "input_layernorm.weight",
        "attention.query_key_value.weight",
        "attention.dense.weight",
        "mlp.dense_h_to_4h.bias",
        "mlp.dense_h_to_4h.weight",
        "mlp.dense_4h_to_h.bias",
        "mlp.dense_4h_to_h.weight",
    ]

    torch.multiprocessing.set_start_method("spawn")
    pool = multiprocessing.Pool(args.processes)
    for name, param in model.named_parameters():
        if name.find("weight") == -1 and name.find("bias") == -1:
            continue
        print(name)
        if name == "transformer.wte.weight":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(
                saved_dir + "model.wte.bin"
            )
        elif name == "transformer.ln_f.bias":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(
                saved_dir + "model.final_layernorm.bias.bin"
            )
        elif name == "transformer.ln_f.weight":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(
                saved_dir + "model.final_layernorm.weight.bin"
            )
        elif name == "lm_head.weight":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(
                saved_dir + "model.lm_head.weight.bin"
            )
        elif name == "lm_head.bias":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(
                saved_dir + "model.lm_head.bias.bin"
            )
        else:
            for i in range(len(huggingface_model_name_pattern)):
                if name.find(huggingface_model_name_pattern[i]) != -1:
                    # Special case for QKV weights
                    if name.find("attn.q_proj.weight") != -1:
                        layer = name.split(".")[2]
                        base_k = f"transformer.h.{layer}."
                        w = model.state_dict()
                        QKV_w = torch.stack(
                            [
                                w[base_k + "attn.q_proj.weight"],
                                w[base_k + "attn.k_proj.weight"],
                                w[base_k + "attn.v_proj.weight"],
                            ]
                        )  # [qkv, n_heads * dim_head, latent_space]
                        QKV_w = QKV_w.permute(2, 0, 1)
                        weights = (
                            QKV_w.detach().cpu().numpy().astype(np_weight_data_type)
                        )
                    else:
                        weights = (
                            param.detach().cpu().numpy().astype(np_weight_data_type)
                        )

                    # Some weights need to be transposed
                    if (
                        name.find("mlp.fc_in.weight") != -1
                        or name.find("mlp.fc_out.weight") != -1
                        or name.find("attn.out_proj.weight") != -1
                    ):
                        weights = weights.T

                    new_name = name.replace("transformer.h.", "layers.").replace(
                        huggingface_model_name_pattern[i], ft_model_name_pattern[i]
                    )

                    pool.starmap(
                        split_and_convert_process,
                        [(0, saved_dir, factor, new_name, weights)],
                    )

    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "-saved_dir", "-o", type=str, help="file name of output file", required=True
    )
    parser.add_argument(
        "-in_file", "-i", type=str, help="HF model name or directory", required=True
    )
    parser.add_argument(
        "-trained_gpu_num",
        "-t_g",
        type=int,
        help="How many gpus for training",
        default=1,
    )
    parser.add_argument(
        "-infer_gpu_num",
        "-i_g",
        type=int,
        help="How many gpus for inference",
        required=True,
    )
    parser.add_argument(
        "-processes",
        "-p",
        type=int,
        help="How many processes to spawn for conversion (default: 4)",
        default=4,
    )
    parser.add_argument(
        "-weight_data_type",
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help="output weight data type",
    )

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print("{}: {}".format(key, vars(args)[key]))
    print("========================================")

    split_and_convert(args)
