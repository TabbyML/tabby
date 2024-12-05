# Code Completion

Code completion is a key feature provided by Tabby in its IDEs/extensions.
Tabby can analyze code repositories or documentation supplied by users
and leverage them to generate helpful code suggestions.

Tabby also allows for more customized configurations by modifying the `[completion]` section in the `config.toml` file.

## Input / Output

This configuration requires tuning of the model serving configuration as well (e.g., context length settings) and can vary significantly based on the model provider (e.g., llama.cpp, vLLM, TensorRT-LLM, etc).
Therefore, please only modify these values after consulting with the model deployment vendor.

```toml
[completion]

# Maximum length of the input prompt, in UTF-8 characters. The default value is set to 1536.
max_input_length = 1536

# Maximum number of decoding tokens. The default value is set to 64.
max_decoding_tokens = 64
```

The default value is set conservatively to accommodate local GPUs and smaller LLMs.

## Additional Language

Tabby supports several built-in programming languages.
For more details, please refer to [Programming Languages](../references/programming-languages.md).

Users can manually configure additional programming languages by defining them in the `config.toml` file.

Below is an example of how to support Swift:

<details>
  <summary>Swift Additional Language Configuration</summary>

```toml title="~/.tabby/config.toml"
[[additional_languages]]
languages = ["swift"]
exts = ["swift"]
line_comment = "//"
top_level_keywords = [
    "import",
    "let",
    "var",
    "func",
    "return",
    "if",
    "else",
    "switch",
    "case",
    "default",
    "break",
    "continue",
    "for",
    "in",
    "while",
    "repeat",
    "guard",
    "throw",
    "throws",
    "do",
    "catch",
    "defer",
    "class",
    "struct",
    "enum",
    "protocol",
    "extension",
    "true",
    "false",
    "nil",
    "self",
    "super",
    "init",
    "deinit",
    "typealias",
    "associatedtype",
    "operator",
    "precedencegroup",
    "inout",
    "async",
    "await",
    "try",
    "rethrows",
    "public",
    "internal",
    "fileprivate",
    "private",
    "open",
    "static",
    "final",
    "dynamic",
    "weak",
    "unowned",
    "lazy",
    "required",
    "optional",
    "convenience",
    "override",
    "mutating",
    "nonmutating",
    "indirect",
    "where",
    "is",
    "as",
    "new",
    "some",
    "Type",
    "Protocol",
    "get",
    "set",
    "willSet",
    "didSet",
    "subscript",
    "fallthrough",
    "Any",
    "Self",
    "unknown",
    "@escaping",
    "@autoclosure",
    "@IBOutlet",
    "@IBAction",
    "@available",
    "@dynamicCallable",
    "@dynamicMemberLookup",
    "@objc",
    "@objcMembers",
    "@propertyWrapper",
    "@main",
    "@resultBuilder",
]
```

</details>