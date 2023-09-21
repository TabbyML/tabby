---
sidebar_position: 4
---

# 🧑‍🔬 Models Directory

We recommend using
* **small models (less than 400M)** for **CPU devices**.
* For **1B to 7B models**, it's advisable to have at least **NVIDIA T4, 10 Series, or 20 Series GPUs**.
* For **7B to 13B models**, we recommend using **NVIDIA V100, A100, 30 Series, or 40 Series GPUs**.

| Model ID                                                               | License                                                                                     |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| [TabbyML/CodeLlama-13B](https://huggingface.co/TabbyML/CodeLlama-13B)  | [Llama2](https://github.com/facebookresearch/llama/blob/main/LICENSE)                       |
| [TabbyML/CodeLlama-7B](https://huggingface.co/TabbyML/CodeLlama-7B)    | [Llama2](https://github.com/facebookresearch/llama/blob/main/LICENSE)                       |
| [TabbyML/StarCoder-7B](https://huggingface.co/TabbyML/StarCoder-7B)    | [BigCode-OpenRAIL-M](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) |
| [TabbyML/StarCoder-3B](https://huggingface.co/TabbyML/StarCoder-3B)    | [BigCode-OpenRAIL-M](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) |
| [TabbyML/StarCoder-1B](https://huggingface.co/TabbyML/StarCoder-1B)    | [BigCode-OpenRAIL-M](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) |
| [TabbyML/SantaCoder-1B](https://huggingface.co/TabbyML/SantaCoder-1B)  | [BigCode-OpenRAIL-M](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) |
| [TabbyML/WizardCoder-3B](https://huggingface.co/TabbyML/WizardCoder-3B)| [OpenRAIL-M](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement)         |
| [TabbyML/WizardCoder-1B](https://huggingface.co/TabbyML/WizardCoder-1B)| [OpenRAIL-M](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement)         |
| [TabbyML/J-350M](https://huggingface.co/TabbyML/J-350M)                | [BSD-3](https://opensource.org/license/bsd-3-clause/)                                       |
| [TabbyML/T5P-220M](https://huggingface.co/TabbyML/T5P-220M)            | [BSD-3](https://opensource.org/license/bsd-3-clause/)                                       |

### CodeLlama-7B / CodeLlama-13B <span title="Apple GPU Support"></span>

Code Llama is a collection of pretrained and fine-tuned generative text models. Theses model is designed for general code synthesis and understanding.

### StarCoder-1B / StarCoder-3B / StarCoder-7B <span title="Apple GPU Support"></span>

StarCoder series model are trained on 80+ programming languages from The Stack (v1.2), with opt-out requests excluded. The model uses Multi Query Attention, a context window of 8192 tokens, and was trained using the Fill-in-the-Middle objective on 1 trillion tokens.

### WizardCoder-1B / WizardCoder-3B <span title="Apple GPU Support"></span>

WizardCoder [(arXiv)](https://arxiv.org/abs/2306.08568) series model are finetuned on StarCoder models with the Evol-Instruct method to adapt to coding tasks. Note that WizardCoder models have used GPT-4 generated data for finetuning, and thus adhere to [OpenAI's limitations](https://openai.com/policies/terms-of-use) for model usage.

### SantaCoder-1B

SantaCoder is the smallest member of the BigCode family of models, boasting just 1.1 billion parameters. This model is specifically trained with a fill-in-the-middle objective, enabling it to efficiently auto-complete function parameters. It offers support for three programming languages: Python, Java, and JavaScript.

### J-350M

Derived from [Salesforce/codegen-350M-multi](https://huggingface.co/Salesforce/codegen-350M-multi), a model of CodeGen family.

### T5P-220M
Derived from [Salesforce/codet5p-220m](https://huggingface.co/Salesforce/codet5p-220m), a model of CodeT5+ family.
