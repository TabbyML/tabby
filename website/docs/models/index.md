---
sidebar_position: 4
---

# 🧑‍🔬 Models Directory

## Completion models (`--model`)

We recommend using

* For **1B to 3B models**, it's advisable to have at least **NVIDIA T4, 10 Series, or 20 Series GPUs**.
* For **7B to 13B models**, we recommend using **NVIDIA V100, A100, 30 Series, or 40 Series GPUs**.

| Model ID                                                              |                                           License                                           | Infilling Support |
| --------------------------------------------------------------------- | :-----------------------------------------------------------------------------------------: | :---------------: |
| [TabbyML/CodeLlama-13B](https://huggingface.co/TabbyML/CodeLlama-13B) |            [Llama2](https://github.com/facebookresearch/llama/blob/main/LICENSE)            |        ✅         |
| [TabbyML/CodeLlama-7B](https://huggingface.co/TabbyML/CodeLlama-7B)   |            [Llama2](https://github.com/facebookresearch/llama/blob/main/LICENSE)            |        ✅         |
| [TabbyML/StarCoder-7B](https://huggingface.co/TabbyML/StarCoder-7B)   | [BigCode-OpenRAIL-M](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) |        ✅         |
| [TabbyML/StarCoder-3B](https://huggingface.co/TabbyML/StarCoder-3B)   | [BigCode-OpenRAIL-M](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) |        ✅         |
| [TabbyML/StarCoder-1B](https://huggingface.co/TabbyML/StarCoder-1B)   | [BigCode-OpenRAIL-M](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) |        ✅         |

## Chat models (`--chat-model`)

To ensure optimal response quality, and given that latency requirements are not stringent in this scenario, we recommend using a model with at least 3B parameters.

| Model ID                                                                |                                       License                                       |
| ----------------------------------------------------------------------- | :---------------------------------------------------------------------------------: |
| [TabbyML/Mistral-7B](https://huggingface.co/TabbyML/Mistral-7B)         |              [Apache 2.0](https://opensource.org/licenses/Apache-2.0)               |
| [TabbyML/WizardCoder-3B](https://huggingface.co/TabbyML/WizardCoder-3B) | [OpenRAIL-M](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) |

## Alternative Registry

By default, Tabby utilizes the [Hugging Face organization](https://huggingface.co/TabbyML) as its model registry. Mainland Chinese users have encountered challenges accessing Hugging Face for various reasons. The Tabby team has established a mirrored at [modelscope](https://www.modelscope.cn/organization/TabbyML), which can be utilized using the following environment variable:

```bash
TABBY_REGISTRY=modelscope tabby serve --model TabbyML/StarCoder-1B
```
