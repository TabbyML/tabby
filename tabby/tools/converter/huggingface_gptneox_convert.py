import argparse
import configparser
import multiprocessing
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import GPTNeoXForCausalLM


def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    else:
        assert False, f"Invalid weight data type {data_type}"


def split_and_convert_process(saved_dir, factor, key, args, config, val):

    if (
        key.find("input_layernorm.weight") != -1
        or key.find("input_layernorm.bias") != -1
        or key.find("post_attention_layernorm.weight") != -1
        or key.find("post_attention_layernorm.bias") != -1
        or key.find("final_layernorm.weight") != -1
        or key.find("final_layernorm.bias") != -1
    ):
        saved_path = saved_dir + f"/model.{key}.bin"
        val.tofile(saved_path)

    elif (
        key.find("attention.dense.bias") != -1
        or key.find("mlp.dense_4h_to_h.bias") != -1
    ):
        saved_path = saved_dir + f"/model.{key}.bin"
        val = (val / factor) if factor > 1 else val
        val.tofile(saved_path)

    else:
        if (
            key.find("attention.dense.weight") != -1
            or key.find("mlp.dense_4h_to_h.weight") != -1
        ):
            split_vals = np.split(val, factor, axis=0)

        elif (
            key.find("mlp.dense_h_to_4h.weight") != -1
            or key.find("mlp.dense_h_to_4h.bias") != -1
        ):
            split_vals = np.split(val, factor, axis=-1)

        elif key.find("attention.query_key_value.bias") != -1:
            local_dim = (int)(val.shape[-1] / 3)
            n_head = config["num_attention_heads"]

            val = val.reshape(n_head, 3, local_dim // n_head)
            val = np.transpose(val, [1, 0, 2]).reshape(3, local_dim)
            split_vals = np.split(val, factor, axis=-1)

        elif key.find("attention.query_key_value.weight") != -1:
            hidden_dim = val.shape[0]
            local_dim = (int)(val.shape[-1] / 3)
            n_head = config["num_attention_heads"]
            # Note that the HF qkv weight are stored as [hidden_size, num_heads, 3, head_hidden]
            # FT needs the shape of [hidden_size, 3, num_heads, head_hidden]
            val = val.reshape(hidden_dim, n_head, 3, local_dim // n_head)
            val = np.transpose(val, [0, 2, 1, 3]).reshape(hidden_dim, 3, local_dim)

            # print(np.mean(np.abs(val[:, 0, :])))
            split_vals = np.split(val, factor, axis=-1)

        else:
            print("[ERROR] cannot find key '{}'".format(key))
            return

        for j in range(factor):
            saved_path = saved_dir + f"/model.{key}.{j}.bin"
            split_vals[j].tofile(saved_path)


def split_and_convert(args):
    saved_dir = args.saved_dir + "/%d-gpu/" % args.infer_gpu_num

    if os.path.exists(saved_dir) == False:
        os.makedirs(saved_dir)

    factor = args.infer_gpu_num

    # load position_embedding from rank 0
    # model = torch.load(ckpt_name)
    model = GPTNeoXForCausalLM.from_pretrained(args.in_file)
    hf_config = vars(model.config)

    np_weight_data_type = get_weight_data_type(args.weight_data_type)

    try:
        model_name = args.model_name
        n_heads = hf_config["num_attention_heads"]
        head_size = hf_config["hidden_size"] // n_heads
        rotary_dim = int(head_size * hf_config["rotary_pct"])
        use_gptj_residual = int(hf_config["use_parallel_residual"])

        config = configparser.ConfigParser()
        config["gptneox"] = {}
        config["gptneox"]["model_name"] = model_name
        config["gptneox"]["head_num"] = str(n_heads)
        config["gptneox"]["size_per_head"] = str(head_size)
        config["gptneox"]["inter_size"] = str(hf_config["intermediate_size"])
        config["gptneox"]["num_layer"] = str(hf_config["num_hidden_layers"])
        config["gptneox"]["rotary_embedding"] = str(rotary_dim)
        config["gptneox"]["vocab_size"] = str(hf_config["vocab_size"])
        config["gptneox"]["start_id"] = str(hf_config["bos_token_id"])
        config["gptneox"]["end_id"] = str(hf_config["eos_token_id"])
        config["gptneox"]["use_gptj_residual"] = str(use_gptj_residual)
        config["gptneox"]["weight_data_type"] = args.weight_data_type

        with open((Path(saved_dir) / f"config.ini").as_posix(), "w") as configfile:
            config.write(configfile)
    except Exception as e:
        print(f"Fail to save the config in config.ini.", e)

    ft_model_name_pattern = [
        "input_layernorm.bias",
        "input_layernorm.weight",
        "attention.query_key_value.bias",
        "attention.query_key_value.weight",
        "attention.dense.bias",
        "attention.dense.weight",
        "post_attention_layernorm.bias",
        "post_attention_layernorm.weight",
        "mlp.dense_h_to_4h.bias",
        "mlp.dense_h_to_4h.weight",
        "mlp.dense_4h_to_h.bias",
        "mlp.dense_4h_to_h.weight",
    ]

    huggingface_model_file_list = [__fn for __fn in os.listdir(args.in_file) if __fn.endswith(".bin")]
    if len(huggingface_model_file_list) > 1:
        multiprocessing_context = multiprocessing.get_context()
        pool_fn = multiprocessing_context.Pool
    else:
        torch.multiprocessing.set_start_method("spawn")
        pool_fn = multiprocessing.Pool

    pool = pool_fn(args.processes)

    for name, param in model.named_parameters():
        array = param.detach().cpu().numpy().astype(np_weight_data_type)
        # print("input shape", name, array.shape)
        if name.find("weight") == -1 and name.find("bias") == -1:
            print("skipped", name)
            continue
        elif name == "gpt_neox.embed_in.weight":
            array.tofile(saved_dir + "model.wte.bin")
        elif name == "gpt_neox.final_layer_norm.bias":
            array.tofile(saved_dir + "model.final_layernorm.bias.bin")
        elif name == "gpt_neox.final_layer_norm.weight":
            array.tofile(saved_dir + "model.final_layernorm.weight.bin")
        elif name == "embed_out.weight":
            array.tofile(saved_dir + "model.lm_head.weight.bin")
        else:
            processed = False
            for i in range(len(ft_model_name_pattern)):
                if name.find(ft_model_name_pattern[i]) != -1:
                    new_name = name.replace("gpt_neox.", "")
                    pool.starmap(
                        split_and_convert_process,
                        [
                            (
                                saved_dir,
                                factor,
                                new_name,
                                args,
                                vars(model.config),
                                array.T,
                            )
                        ],
                    )
                    processed = True
                    break

            if not processed:
                print("Unused layer", name)

    pool.close()
    pool.join()

    # Post-process biases if use_gptj_residual is True
    if use_gptj_residual:
        for layer_idx in range(hf_config["num_hidden_layers"]):
            attn_bias = np.fromfile(
                saved_dir + f"/model.layers.{layer_idx}.attention.dense.bias.bin",
                dtype=np_weight_data_type,
            )
            mlp_bias = np.fromfile(
                saved_dir + f"/model.layers.{layer_idx}.mlp.dense_4h_to_h.bias.bin",
                dtype=np_weight_data_type,
            )

            (attn_bias + mlp_bias).astype(np_weight_data_type).tofile(
                saved_dir + f"/model.layers.{layer_idx}.mlp.attention.bias.sum.bin"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "-saved_dir", "-o", type=str, help="file name of output file", required=True
    )
    parser.add_argument(
        "-in_file",
        "-i",
        type=str,
        help="file name of input checkpoint file",
        required=True,
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
        "-weight_data_type", type=str, default="fp32", choices=["fp32", "fp16"]
    )
    parser.add_argument(
        "-model_name", "-m_n", type=str, help="model name", required=True
    )

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print("{}: {}".format(key, vars(args)[key]))
    print("========================================")

    __dir = os.path.join(args.saved_dir, "%d-gpu" % args.infer_gpu_num)
    assert not os.path.exists(__dir), "target path has exist, please remove %s first." % __dir

    split_and_convert(args)