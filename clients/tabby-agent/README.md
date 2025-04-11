# Tabby Agent

The [tabby-agent](https://www.npmjs.com/package/tabby-agent) is an agent used for communication with the [Tabby](https://www.tabbyml.com) server. It is based on Node.js v18 and runs as a language server.

**Breaking Changes**: The tabby-agent will only support running as a language server since version 1.7.0.

The tabby-agent mainly supports the following features of LSP:

- Completion (textDocument/completion)
- Inline Completion (textDocument/inlineCompletion, since LSP v3.18.0)

For collecting more context to enhance the completion quality or providing more features like inline chat editing, the tabby-agent extends the protocol with some custom methods starting with `tabby/*`. These methods are used in Tabby-provided editor extensions.

## Usage

**Note**: For VSCode, IntelliJ Platform IDEs, and Vim/NeoVim, it is recommended to use the Tabby-provided extensions, which run the Tabby Agent underlying.

- [VSCode](https://marketplace.visualstudio.com/items?itemName=TabbyML.vscode-tabby)
- [IntelliJ Platform IDEs](https://plugins.jetbrains.com/plugin/22379-tabby)
- [Vim/NeoVim](https://github.com/TabbyML/vim-tabby)

The following guide is only for users who want to set up the tabby-agent as a language server manually.

### Start the Language Server

```bash
npx tabby-agent --stdio
```

### Connect the IDE to the tabby-agent

Since most text editors have their built-in LSP clients or popular LSP client plugins, you can easily connect to the tabby-agent from your editor. Here are some example configurations for popular editors.

#### Vim/Neovim (coc.nvim)

There are several Vim plugins that provide LSP support. One of them is [coc.nvim](https://github.com/neoclide/coc.nvim). To use the tabby-agent as a language server, you can add the following code to your `:CocConfig`.

```json
{
  "languageserver": {
    "tabby-agent": {
      "command": "npx",
      "args": ["tabby-agent", "--stdio"],
      "filetypes": ["*"]
    }
  }
}
```

#### Emacs

The package [lsp-mode](https://github.com/emacs-lsp/lsp-mode) provides an LSP client for Emacs. You can add the following code to your Emacs configuration script to use the tabby-agent as a language server.

```emacs-lisp
(with-eval-after-load 'lsp-mode
  (lsp-register-client
    (make-lsp-client  :new-connection (lsp-stdio-connection '("npx" "tabby-agent" "--stdio"))
                      ;; you can select languages to enable Tabby language server
                      :activation-fn (lsp-activate-on "typescript" "javascript" "toml")
                      :priority 1
                      :add-on? t
                      :server-id 'tabby-agent)))
```

#### Helix

[Helix](https://helix-editor.com/) has built-in LSP support. To use the tabby-agent as a language server, you can add the following code to your `languages.toml`.

```toml
[language-server.tabby]
command = "npx"
args = ["tabby-agent", "--stdio"]

# Add Tabby as the second language server for your specific languages
[[language]]
name = "typescript"
language-servers = ["typescript-language-server", "tabby"]

[[language]]
name = "toml"
language-servers = ["taplo", "tabby"]
```

#### More Editors

You are welcome to contribute by adding example configurations for your favorite editor. Please submit a PR with your additions.

### Configurations

Please refer to the [configuration documentation](https://tabby.tabbyml.com/docs/extensions/configurations/) for more details.

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
