# v0.4.0 [Unreleased]

## Features
* Supports golang: https://github.com/TabbyML/tabby/issues/553

## Fixes and Improvements

# v0.3.1
## Fixes and improvements

* Fix GPU OOM issue caused the parallelism: https://github.com/TabbyML/tabby/issues/541, https://github.com/TabbyML/tabby/issues/561
* Fix git safe directory check in docker: https://github.com/TabbyML/tabby/issues/569

# v0.3.0

## Features
### Retrieval-Augmented Code Completion Enabled by Default

The currently supported languages are:

* Rust
* Python
* JavaScript / JSX
* TypeScript / TSX

A blog series detailing the technical aspects of Retrieval-Augmented Code Completion will be published soon. Stay tuned!

## Fixes and Improvements

* Fix [Issue #511](https://github.com/TabbyML/tabby/issues/511) by marking ggml models as optional.
* Improve stop words handling by combining RegexSet into Regex for efficiency.

# v0.2.2 (10/09/2023)
## Fixes and improvements

* Fix a critical issue that might cause request dead locking in ctranslate2 backend (when loading is heavy)

# v0.2.1 (10/03/2023)
## Features
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

## Fixes and improvements

* Implemented more accurate UTF-8 incremental decoding in the [GitHub pull request](https://github.com/TabbyML/tabby/pull/491).
* Fixed the stop words implementation by utilizing RegexSet to isolate the stop word group.
* Improved model downloading logic; now Tabby will attempt to fetch the latest model version if there's a remote change, and the local cache key becomes stale.
* set default num_replicas_per_device for ctranslate2 backend to increase parallelism.
