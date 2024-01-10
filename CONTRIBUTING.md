# Contributing

Thank you for your interest in contributing to Tabby! We appreciate all contributions. For a better experience and support, join us on [Slack](https://links.tabbyml.com/join-slack)!

To begin contributing to Tabby, first clone the repository locally:

```
git clone --recurse-submodules https://github.com/TabbyML/tabby
```

If you have already cloned the repository, you can initialize submodules with this command:

```
git submodule update --recursive --init
```

Make sure you have installed [Rust](https://www.rust-lang.org/learn/get-started) and the following dependencies for MacOS or Debian-based Linux distributions:

```bash
# For MacOS
brew install protobuf

# For Ubuntu / Debian
apt-get install protobuf-compiler libopenblas-dev
```

Before proceeding, ensure that all tests are passing locally:

```
cargo test -- --skip golden
```

This will help ensure everything is working correctly and avoid surprises with local breakages.

## Building and Running

Tabby can be run through cargo in much the same manner as docker:

```
cargo run serve --model TabbyML/StarCoder-1B
```

This will run Tabby locally on CPU, which is not optimal for performance. Depending on your GPU and its compatibility, you may be able to run Tabby with GPU acceleration. Please make sure you have CUDA or ROCm installed, for Nvidia or AMD graphics cards respectively.

To run Tabby locally with CUDA (Nvidia):

```
cargo run --release --features cuda serve --model TabbyML/StarCoder-1B
```

To run Tabby locally with ROCm (AMD):

```
cargo run --release --features rocm serve --model TabbyML/StarCoder-1B
```

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
- `ee/tabby-webserver` - The webserver for Tabby with privilege management and a chatbot playground. Also includes GraphQL API implementation. Must use `--webserver` on CLI to enable
- `ee/tabby-db` - The database backing the webserver
- `ee/tabby-ui` - Frontend for the Tabby webserver
