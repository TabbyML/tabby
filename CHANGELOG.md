# v0.8.0 [Unreleased]

## Features

## Fixes and Improvements

* Add windows cpu binary distribution.
* Add linux rocm (AMD GPU) binary distribution.
* Fix cached permanent redirection on certain browsers(e.g chrome) when `--webserver` is not enabled.

# v0.7.0 (12/15/2023)

## Features

* Tabby now includes built-in user management and secure access, ensuring that it is only accessible to your team.
* The `--webserver` flag is a new addition to `tabby serve` that enables secure access to the tabby server. When this flag is on, IDE extensions will need to provide an authorization token to access the instance.
  - Some functionalities that are bound to the webserver (e.g. playground) will also require the `--webserver` flag.


## Fixes and Improvements

*  Fix https://github.com/TabbyML/tabby/issues/1036, events log should be written to dated json files.

# v0.6.0 (11/27/2023)

## Features

* Add distribution support (running completion / chat model on different process / machine).
* Add conversation history in chat playground.
* Add `/metrics` endpoint for prometheus metrics collection. 

## Fixes and Improvements

* Fix the slow repository indexing due to constraint memory arena in tantivy index writer.
* Make `--model` optional, so users can create a chat only instance.
* Add `--parallelism` to control the throughput and VRAM usage: https://github.com/TabbyML/tabby/pull/727

# v0.5.5 (11/09/2023)

## Fixes and Improvements

## Notice

* llama.cpp backend (CPU, Metal) now requires a redownload of gguf model due to upstream format changes: https://github.com/TabbyML/tabby/pull/645 https://github.com/ggerganov/llama.cpp/pull/3252
* Due to indexing format changes, the `~/.tabby/index` needs to be manually removed before any further runs of `tabby scheduler`.
* `TABBY_REGISTRY` is replaced with `TABBY_DOWNLOAD_HOST` for the github based registry implementation.

## Features

* Improved dashboard UI.

## Fixes and Improvements

* Cpu backend is switched to llama.cpp: https://github.com/TabbyML/tabby/pull/638
* add `server.completion_timeout` to control the code completion interface timeout: https://github.com/TabbyML/tabby/pull/637
* Cuda backend is switched to llama.cpp: https://github.com/TabbyML/tabby/pull/656
* Tokenizer implementation is switched to llama.cpp, so tabby no longer need to download additional tokenizer file: https://github.com/TabbyML/tabby/pull/683
* Fix deadlock issue reported in https://github.com/TabbyML/tabby/issues/718

# v0.4.0 (10/24/2023)

## Features

* Supports golang: https://github.com/TabbyML/tabby/issues/553
* Supports ruby: https://github.com/TabbyML/tabby/pull/597
* Supports using local directory for `Repository.git_url`: use `file:///path/to/repo` to specify a local directory.
* A new UI design for webserver.

## Fixes and Improvements

* Improve snippets retrieval by dedup candidates to existing content + snippets: https://github.com/TabbyML/tabby/pull/582

# v0.3.1 (10/21/2023)
## Fixes and improvements

* Fix GPU OOM issue caused the parallelism: https://github.com/TabbyML/tabby/issues/541, https://github.com/TabbyML/tabby/issues/587
* Fix git safe directory check in docker: https://github.com/TabbyML/tabby/issues/569

# v0.3.0 (10/13/2023)

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
