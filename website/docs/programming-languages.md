---
sidebar_position: 6
---

# 🧑‍💻 Programming Languages

Most models nowadays support a large number of programming languages (thanks to [The Stack](https://huggingface.co/datasets/bigcode/the-stack), which has collected 358 programming languages).
In Tabby, we need to add configuration for each language to maximize performance and completion quality.

Currently, there are two aspects of support that need to be added for each language.

**Stop Words**

Stop words determine when the language model can early stop its decoding steps, resulting in better latency and affecting the quality of completion. We suggest adding all top-level keywords as part of the stop words.

**Repository Context**

We parse languages into chunks and compute a token-based index for serving time Retrieval Augmented Code Completion. In Tabby, we define these repository contexts as [treesitter queries](https://tree-sitter.github.io/tree-sitter/using-parsers#query-syntax), and the query results will be indexed.

For an actual example of an issue or pull request adding the above support, please check out https://github.com/TabbyML/tabby/issues/553 as a reference.

## Supported Languages

* [Rust](https://www.rust-lang.org/)
* [Python](https://www.python.org/)
* [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
* [TypeScript](https://www.typescriptlang.org/)

## Languages Missing Certain Support

| Language | Stop Words (time to contribute: <5 min) | Repository Context (time to contribute: <1 hr) |
| :------: | :-------------------------------------: | :--------------------------------------------: |
|  C/C++   |                    🚫                    |                       🚫                        |
|    C#    |                    🚫                    |                       🚫                        |
|   CSS    |                    🚫                    |                       🚫                        |
|    Go    |                    🚫                    |                       🚫                        |
| Haskell  |                    🚫                    |                       🚫                        |
|   Java   |                    🚫                    |                       🚫                        |
|  Julia   |                    🚫                    |                       🚫                        |
|   Lua    |                    🚫                    |                       🚫                        |
|   PHP    |                    🚫                    |                       🚫                        |
|   Perl   |                    🚫                    |                       🚫                        |
|   Ruby   |                    🚫                    |                       🚫                        |
|  Scala   |                    🚫                    |                       🚫                        |