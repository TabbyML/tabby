# [Unreleased]

## 🚀 Features
## 🧰 Fixes and improvements

# v0.2.0 (10/03/2023)
## 🚀 Features
### Chat Model & Web Interface

We have introduced a new argument, `--chat-model`, which allows you to specify the model for the chat playground located at http://localhost:8080/playground

To utilize this feature, use the following command in the terminal:

```bash
tabby serve --device metal --model TabbyML/StarCoder-1B --chat-model TabbyML/Mistral-7B
```

### ModelScope Model Registry

Mainland Chinese users have been facing challenges accessing Hugging Face due to various reasons. The Tabby team is actively working to address this issue by mirroring models to a hosting provider in mainland China called modelscope.cn.

```bash
# Download from the Modelscope registry
TABBY_REGISTRY=modelscope tabby download --model TabbyML/WizardCoder-1B
```

## 🧰 Fixes and improvements

* Implemented more accurate UTF-8 incremental decoding in the [GitHub pull request](https://github.com/TabbyML/tabby/pull/491).
* Fixed the stop words implementation by utilizing RegexSet to isolate the stop word group.
* Improved model downloading logic; now Tabby will attempt to fetch the latest model version if there's a remote change, and the local cache key becomes stale.
