---
sidebar_position: 4
---

# 🧑‍🔬 Models Directory

We maintains a recommended collection of models varies from 200M to 10B+ for various use cases.

| Model ID                                                              | License                                                                                     | <span title="Apple M1/M2 Only">Metal Support</span> |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | :-------------------------------------------------: |
| [TabbyML/CodeLlama-13B](https://huggingface.co/TabbyML/CodeLlama-13B) | [Llama2](https://github.com/facebookresearch/llama/blob/main/LICENSE)                       |                         ❌                          |
| [TabbyML/CodeLlama-7B](https://huggingface.co/TabbyML/CodeLlama-7B)   | [Llama2](https://github.com/facebookresearch/llama/blob/main/LICENSE)                       |                         ✅                          |
| [TabbyML/StarCoder-1B](https://huggingface.co/TabbyML/StarCoder-1B)   | [BigCode-OpenRAIL-M](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) |                         ❌                          |
| [TabbyML/SantaCoder-1B](https://huggingface.co/TabbyML/SantaCoder-1B) | [BigCode-OpenRAIL-M](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) |                         ❌                          |
| [TabbyML/J-350M](https://huggingface.co/TabbyML/J-350M)               | [BSD-3](https://opensource.org/license/bsd-3-clause/)                                       |                         ❌                          |
| [TabbyML/T5P-220M](https://huggingface.co/TabbyML/T5P-220M)           | [BSD-3](https://opensource.org/license/bsd-3-clause/)                                       |                         ❌                          |

### CodeLlama-7B / CodeLlama-13B
Code Llama is a collection of pretrained and fine-tuned generative text models. Theses model is designed for general code synthesis and understanding.

### StarCoder-1B
StarCoderBase-1B is a 1B parameter model trained on 80+ programming languages from The Stack (v1.2), with opt-out requests excluded. The model uses Multi Query Attention, a context window of 8192 tokens, and was trained using the Fill-in-the-Middle objective on 1 trillion tokens.

### SantaCoder-1B
SantaCoder is the smallest member of the BigCode family of models, boasting just 1.1 billion parameters. This model is specifically trained with a fill-in-the-middle objective, enabling it to efficiently auto-complete function parameters. It offers support for three programming languages: Python, Java, and JavaScript.

### J-350M
Derived from [Salesforce/codegen-350M-multi](https://huggingface.co/Salesforce/codegen-350M-multi), a model of CodeGen family.

### T5P-220M
Derived from [Salesforce/codet5p-220m](https://huggingface.co/Salesforce/codet5p-220m), a model of CodeT5+ family.