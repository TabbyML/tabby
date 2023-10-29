---
authors: [ meng, gyxlucy ]

tags: [release]

---
# Announcing our $3.2M seed round, and the long-awaited RAG release in Tabby v0.3.0

We are excited to announce that TabbyML has raised a [$3.2M seed round](https://techcrunch.com/2023/10/10/tabbyml-github-copilot-alternative-raises-3-2-million/) to move towards our goal of building an open ecosystem to supercharge developer experience with LLM 🎉🎉🎉. 

## Why Tabby 🐾 ? 
With over 10 years coding experience, we recognize the transformative potential of LLMs in reshaping developer toolchains. While many existing products lean heavily on cloud-based end-to-end solutions, we firmly believe that for AI to be genuinely the core of every developer's toolkit, the next-gen LLM-enhanced developer tools should embrace an open ecosystem. This approach promotes not just flexibility for easy customization, but also fortifies security.

Today, Tabby stands out as the most popular and user-friendly solution to enable coding assistant experience fully owned by users. Looking ahead, we are poised to delve even further into the developer lifecycle, and innovate across the full spectrum. At TabbyML, developers aren't just participants — they are at the heart of the LLM revolution.


## Release v0.3.0 - Retrieval Augmented Code Completion 🎁
Tabby also comes to a [v0.3.0 release](https://github.com/TabbyML/tabby/releases/tag/v0.3.0), with the support of retrieval-augmented code completion enabled by default. Enhanced by repo-level retrieval, Tabby gets smarter at your codebase and will quickly reference to a related function / code example from another file in your repository.

A blog series detailing the technical designs of retrieval-augmented code completion will be published soon. Stay tuned!🔔

***Example prompt for retrieval-augmented code completion:***

```rust
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
        .
```
