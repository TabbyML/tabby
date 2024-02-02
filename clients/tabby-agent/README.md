# Tabby Agent

The Tabby Agent is an agent used for communication with the Tabby server. It is based on Node.js v18 and implements several common features to enhance code completion, including caching, debouncing, and post-processing.

This package exports both libraries and a CLI tool.

## Language Server

To run the agent as a language server, you can pass the argument `--lsp` and an additional IO option to the CLI tool. The available IO options are `--stdio`, `--node-ipc` or `--socket=<PORT>`.

```bash
npx tabby-agent --lsp --stdio
```

The language server provides the following features:

- Completion
- Inline Completion (WIP, upcoming LSP v3.18 feature)

Since most text editors have their built-in LSP clients or popular LSP client plugins, you can easily connect to the Tabby agent language server from your editor. Here are some example configurations for popular editors.

### VSCode

A common way to add a new language server to VSCode is by creating a new extension. We provide an example extension for VSCode in [example-vscode-lsp](https://github.com/tabbyml/tabby/blob/master/clients/example-vscode-lsp).

**Note**: Tabby provides an official extension for VSCode, you can install it from [marketplace](https://marketplace.visualstudio.com/items?itemName=tabbyml.vscode-tabby). The example provided here is just for reference on how to use the Tabby agent as a language server.

### Emacs

The package [lsp-mode](https://github.com/emacs-lsp/lsp-mode) provides an LSP client for Emacs. You can add the following code to your Emacs configuration script to use the Tabby agent as a language server.

```emacs-lisp
(with-eval-after-load 'lsp-mode
  (lsp-register-client
    (make-lsp-client  :new-connection (lsp-stdio-connection '("npx" "tabby-agent" "--lsp" "--stdio"))
                      ;; you can select languages to enable Tabby language server
                      :activation-fn (lsp-activate-on "typescript" "javascript" "toml")
                      :priority 1
                      :add-on? t
                      :server-id 'tabby-agent)))
```

### Vim/Neovim

There are several Vim/Neovim plugins that provide LSP support. One of them is [coc.nvim](https://github.com/neoclide/coc.nvim). To use the Tabby agent as a language server, you can add the following code to your :CocConfig.

```json
{
  "languageserver": {
    "tabby-agent": {
      "command": "npx",
      "args": ["tabby-agent", "--lsp", "--stdio"],
      "filetypes": ["*"]
    }
  }
}
```

### Helix

[Helix](https://helix-editor.com/) has built-in LSP support. To use the Tabby agent as a language server, you can add the following code to your `languages.toml`.

```toml
[language-server.tabby]
command = "npx"
args = ["tabby-agent", "--lsp", "--stdio"]

# Add Tabby as the second language server for your specific languages
[[languages]]
name = "typescript"
language-servers = ["typescript-language-server", "tabby"]

[[languages]]
name = "toml"
language-servers = ["taplo", "tabby"]
```

### More Editors

You are welcome to contribute by adding example configurations for your favorite editor. Please submit a PR with your additions.

## License

Copyright (c) 2023-2024 TabbyML, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
