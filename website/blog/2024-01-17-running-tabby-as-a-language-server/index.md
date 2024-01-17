---
slug: running-tabby-as-a-language-server
title: "Running Tabby as a Language Server"
authors: []
tags: []
---

We are excited to announce that we have added a new feature to Tabby, allowing you to run it as a [language server](https://microsoft.github.io/language-server-protocol/).

Previously, we provided extensions for popular editors like [VSCode](https://marketplace.visualstudio.com/items?itemName=TabbyML.vscode-tabby), [IntelliJ Platform IDEs](https://plugins.jetbrains.com/plugin/22379-tabby), and [VIM/NeoVIM](https://github.com/TabbyML/vim-tabby), which offered inline code completion. These plugins were all based on the same client agent called [tabby-agent](https://github.com/tabbyml/tabby/tree/main/clients/tabby-agent/), which implemented features such as caching, debouncing, and post-processing to enhance code completion. The tabby-agent communicated with the IDE via a customized protocol based on JSON Lines, originally designed for compatibility with VIM's JSON mode channel. By adding support for the Language Server Protocol, we should make Tabby more flexible and welcoming to contributors.

## Running Tabby as a Language Server

To run Tabby as a language server, you'll first need to follow [this guide](https://tabby.tabbyml.com/docs/installation/) to set up your Tabby server.

We have released tabby-agent as a separate package on [npm](https://www.npmjs.com/package/tabby-agent), making it easy to install and run as a language server. To get started, ensure that you have Node.js v18 or above installed, and then execute the following command:

```bash
npx tabby-agent --lsp --stdio
```

You can configure the settings of tabby-agent by editing the config file located at `~/.tabby-client/agent/config.toml`. See [this documentation](https://tabby.tabbyml.com/docs/extensions/configurations) for more details on these configurations.

The language server provides code completion functionality using the standard `textDocument/completion` protocol. It can suggest code completions based on the context of the code, whether it's a line or a block, rather than just a single word.

It's worth mentioning that there is a proposed protocol `textDocument/inlineCompletion` in the LSP Specification upcoming version 3.18, which would be more suitable for these cases, we look forward to its inclusion in future versions.

## Connect Your Editor to Tabby

Since most text editors support built-in LSP clients or popular LSP client plugins, you can easily connect to the Tabby agent language server from your editor. Let's use NeoVim and [coc.nvim](https://github.com/neoclide/coc.nvim) as an example to show you how to configure your editor to connect to Tabby.

After installing coc.nvim, use the `:CocConfig` command to edit the configuration file. Add the following configuration to your file:

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

Save the file and restart your editor. Open a file and start typing some code to see the code completion suggestions from Tabby.

![coc-tabby-completion](coc-tabby-completion.png)

For more examples, please refer to [this documentation](https://github.com/tabbyml/tabby/tree/main/clients/tabby-agent/). If you'd like to share configurations for your favorite editors, feel free to submit a pull request!

## Create a Plugin for a New Editor

In the previous examples, we showed how to make Tabby suggest code completions in the traditional completion list. However, this method may not be suitable for displaying multi-line code completions. As most LSP clients do not yet support inline completion, you may want to create a plugin for an editor that provides inline completion.
We have provided an example project [here](https://github.com/tabbyml/tabby/tree/main/clients/example-vscode-lsp) to demonstrate how to communicate with Tabby via LSP.

The language server support is still in its first version, and your feedback will be invaluable in making it even better. Please feel free to create an issue or join our [Slack community](https://links.tabbyml.com/join-slack) to share your ideas. 

Happy coding!
