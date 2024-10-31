# 🤝 Contributing

Thank you for your interest in contributing to Tabby! We appreciate all contributions. For a better experience and support, join us on [Slack](https://links.tabbyml.com/join-slack)!

## Local Setup

To begin contributing to Tabby, first clone the repository locally:

```
git clone --recurse-submodules https://github.com/TabbyML/tabby
```

If you have already cloned the repository, you can initialize submodules with this command:

```
git submodule update --recursive --init
```

Make sure you have installed [Rust](https://www.rust-lang.org/learn/get-started), and one of the following dependencies may need to be installed depending on your system:

```bash
# For MacOS
brew install protobuf

# For Ubuntu / Debian
apt-get install protobuf-compiler libopenblas-dev

# For Windows 11 with Chocolatey package manager
choco install protoc
```

Some of the tests require mailpit SMTP server which you can install following this [instruction](https://github.com/axllent/mailpit?tab=readme-ov-file#installation)

Before proceeding, ensure that all tests are passing locally:

```
cargo test -- --skip golden
```

This will help ensure everything is working correctly and avoid surprises with local breakages.

Golden tests, which run models and check their outputs against previous "golden snapshots", should be skipped for most development purposes, as they take a very long time to run (especially the tests running the models on CPU). You may still want to run them if your changes relate to the functioning of or integration with the generative models, but skipping them is recommended otherwise.

Optionally, to use a GPU make sure you have the correct drivers and libraries installed for your device:

> **CUDA for Nvidia** - [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
> **ROCm for AMD** - [Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html), [Windows](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)

## Building and Running

Tabby can be run through `cargo` in much the same manner as docker:

```
cargo run serve --model TabbyML/StarCoder-1B
```

This will run Tabby locally on CPU, which is not optimal for performance. Depending on your GPU and its compatibilities, you may be able to run Tabby with GPU acceleration. First insure you have the dependencies for your Nvidia or AMD GPU installed. No extra library installation is necessary for Apple silicon (M1/M2) using Metal.

To run Tabby locally with CUDA (NVIDIA):

```
cargo run --features cuda serve --model TabbyML/StarCoder-1B --device cuda
```

To run Tabby locally with ROCm (AMD):

```
cargo run --features rocm serve --model TabbyML/StarCoder-1B --device rocm
```

To run Tabby locally with Vulkan:

```
cargo run --features vulkan serve --model TabbyML/StarCoder-1B --device vulkan
```

To run Tabby locally with Metal (Apple M1/M2):

```
cargo run serve --model TabbyML/StarCoder-1B --device metal
```

After running the respective command, you should see an output similar to the below (after compilation). The demonstration is for ROCm (AMD).

![image](https://github.com/TabbyML/tabby/assets/14198267/8f21d495-882d-462c-b426-7c495f38a5d8)

By default, Tabby will start on `localhost:8080` and serve requests.

## Project Layout

Tabby is broken up into several crates, each responsible for a different part of the functionality. These crates fall into two categories: Fully open source features, and enterprise features. All open-source feature crates are located in the `/crates` folder in the repository root, and all enterprise feature crates are located in `/ee`.

### Crates

- `crates/tabby` - The core tabby application, this is the main binary crate defining CLI behavior and driving the API
- `crates/tabby-common` - Interfaces and type definitions shared across most other tabby crates, especially types used for serialization
- `crates/tabby-download` - Very small crate, responsible for downloading models at runtime
- `crates/tabby-scheduler` - Defines jobs that need to run periodically for syncing and indexing code
- `crates/tabby-inference` - Defines interfaces for interacting with text generation models
- `crates/llama-cpp-bindings` - Raw bindings to talk with the actual models in C++ from Rust
- `ee/tabby-webserver` - The webserver for Tabby with privilege management and a chatbot playground. Also includes GraphQL API implementation.
- `ee/tabby-db` - The database backing the webserver
- `ee/tabby-ui` - Frontend for the Tabby webserver

## Picking an Issue

This [search filter](https://github.com/TabbyML/tabby/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22+no%3Aassignee) will show all the issues currently marked as "open" and "good first issue" that aren't currently assigned to anyone.
Any of them would be a good choice for starting out, and choosing one that already has some conversation may help give context and ensure it's relevant.

Most issues will have a link to the related location in the code, and if they don't, you can always reach out to us on Slack or mention one of us in the issue to provide more context.

## Code Review

You can feel free to open PRs that aren't quite ready yet, to work on them. If you do this, please make sure to [mark the pull request as a draft](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/changing-the-stage-of-a-pull-request).

Once your PR is ready, please request review from one of the [Tabby team members](https://github.com/orgs/TabbyML/people), and watch for replies asking for any changes. Once approved, you can merge your code into Tabby!

# Changelog

Tabby used [changie](https://changie.dev/) to track unreleased features, it's preferred the changelog is added as part of implementation pr. To create an unreleased feature, use `changie new` command.

# Contributing to Docs

To begin contributing to Tabby's docs website, make sure you installed node lts and yarn:

```
cd website
yarn install
yarn start
```