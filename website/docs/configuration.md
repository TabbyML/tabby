---
sidebar_position: 5
---

# ⚙️ Configuration

Tabby server will look for a configuration file at `~/.tabby/config.toml` for advanced features.

### Repository context for code completion

To enable repository level context for code completion, you can add the following to your configuration file:

```toml title="~/.tabby/config.toml"
# Index three repositories' source code as additional context for code completion.

[[repositories]]
name = "tabby"
git_url = "https://github.com/TabbyML/tabby.git"

# git through ssh protocol.
[[repositories]]
name = "CTranslate2"
git_url = "git@github.com:OpenNMT/CTranslate2.git"

# local directory is also supported!
[[repositories]]
name = "repository_a"
git_url = "file:///home/users/repository_a"
```

Once this is set, you can run `tabby scheduler` to index the source code repository.

:::tip
By default, `tabby scheduler` runs in a daemon and processes its pipeline every 5 hours. To run the pipeline immediately, use `tabby scheduler --now`.
:::

```bash title="artifacts produced by tabby scheduler"
~/.tabby % ls dataset
data.jsonl

~/.tabby % ls index
1a8729fa34d844df984b444f4def1456.fast      2ed712d4a7a44ed797dd4ff5ceaf4312.fieldnorm
b42ca53fe6f94d0c8e96f947318278ba.idx       1a8729fa34d844df984b444f4def1456.fieldnorm 
2ed712d4a7a44ed797dd4ff5ceaf4312.idx       b42ca53fe6f94d0c8e96f947318278ba.pos
...
```

In a code completion request, additional context from the source code repository will be attached to the prompt for better completion quality. For example:

```rust title="Example prompt for code completion, with retrieval augmented enabled"
// Path: crates/tabby/src/serve/engine.rs
// fn create_llama_engine(model_dir: &ModelDir) -> Box<dyn TextGeneration> {
//     let options = llama_cpp_bindings::LlamaEngineOptionsBuilder::default()
//         .model_path(model_dir.ggml_q8_0_file())
//         .tokenizer_path(model_dir.tokenizer_file())
//         .build()
//         .unwrap();
//
//     Box::new(llama_cpp_bindings::LlamaEngine::create(options))
// }
//
// Path: crates/tabby/src/serve/engine.rs
// create_local_engine(args, &model_dir, &metadata)
//
// Path: crates/tabby/src/serve/health.rs
// args.device.to_string()
//
// Path: crates/tabby/src/serve/mod.rs
// download_model(&args.model, &args.device)
    } else {
        create_llama_engine(model_dir)
    }
}

fn create_ctranslate2_engine(
    args: &crate::serve::ServeArgs,
    model_dir: &ModelDir,
    metadata: &Metadata,
) -> Box<dyn TextGeneration> {
    let device = format!("{}", args.device);
    let options = CTranslate2EngineOptionsBuilder::default()
        .model_path(model_dir.ctranslate2_dir())
        .tokenizer_path(model_dir.tokenizer_file())
        .device(device)
        .model_type(metadata.auto_model.clone())
        .device_indices(args.device_indices.clone())
        .build()
        .⮹
```

## Usage Collection
Tabby collects usage stats by default. This data will only be used by the Tabby team to improve its services.

### What data is collected?
We collect non-sensitive data that helps us understand how Tabby is used. For now we collects `serve` command you used to start the server.
As of the date 10/07/2023, the following information has been collected:

```rust
struct HealthState {
    model: String,
    chat_model: Option<String>,
    device: String,
    arch: String,
    cpu_info: String,
    cpu_count: usize,
    cuda_devices: Vec<String>,
    version: Version,
}
```

For an up-to-date list of the fields we have collected, please refer to [health.rs](https://github.com/TabbyML/tabby/blob/main/crates/tabby/src/serve/health.rs#L11).

### How to disable it
To disable usage collection, set the `TABBY_DISABLE_USAGE_COLLECTION` environment variable by `export TABBY_DISABLE_USAGE_COLLECTION=1`.
