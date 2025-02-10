---
sidebar_position: 6
---

# Programming Languages

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
* [Golang](https://go.dev/)
* [Ruby](https://www.ruby-lang.org/)
* [Java](https://www.java.com/)
* [Kotlin](https://www.kotlinlang.org/)
* [C/C++](https://cplusplus.com/)
* [PHP](https://www.php.net/)
* [C#](https://learn.microsoft.com/en-us/dotnet/csharp/)
* [Solidity](https://soliditylang.org/)
* [R](https://www.r-project.org/)
* [Dart](https://dart.dev/)
* [Lua](https://www.lua.org)
* [Elixir](https://elixir-lang.org)
* [OCaml](https://ocaml.org/)
* [GDScript](https://gdscript.com/)

## Languages Missing Certain Support

| Language | Stop Words (time to contribute: ~5 min) | Repository Context (time to contribute: ~1 hr) |
| :------: | :-------------------------------------: | :--------------------------------------------: |
|   CSS    |                    🚫                    |                       🚫                        |
| Haskell  |                    🚫                    |                       🚫                        |
|  Julia   |                    🚫                    |                       🚫                        |
|   Perl   |                    🚫                    |                       🚫                        |
|  Scala   |                    🚫                    |                       🚫                        |
