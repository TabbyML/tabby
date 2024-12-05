# Tabby Plugin for Vim and Neovim

Tabby is a self-hosted AI coding assistant that can suggest multi-line code or full functions in real-time. For more information, please check out our [website](https://tabbyml.com/) and [GitHub](https://github.com/TabbyML/tabby).  
If you encounter any problems or have any suggestions, please [open an issue](https://github.com/TabbyML/tabby/issues/new) or join our [Slack community](https://links.tabbyml.com/join-slack) for support.

## Notable Changes in vim-tabby Plugin 2.0

Since version 2.0, the vim-tabby plugin is designed as two parts:
1. **LSP Client Extension**:
   - Relies on an LSP client and extends it with methods (such as `textDocument/inlineCompletion`) to communicate with the tabby-agent.
   - Note: The Node.js script of tabby-agent is no longer a built-in part of the vim-tabby plugin. You need to install tabby-agent separately via npm, and the LSP client will launch it using the command `npx tabby-agent --stdio`.
2. **Inline Completion UI**:
   - Automatically triggers inline completion requests when typing.
   - Renders the inline completion text as ghost text.
   - Sets up actions with keyboard shortcuts to accept or dismiss the inline completion.

## Requirements

The Tabby plugin requires the following dependencies:

- **Tabby Server**: The backend LLM server. You can install the Tabby server locally or have it hosted on a remote server. For Tabby server installation, please refer to this [documentation](https://tabby.tabbyml.com/docs/installation/).
- **Tabby Agent (LSP server)**: Requires [Node.js](https://nodejs.org/en/download/) version v18.0+ and [tabby-agent](https://www.npmjs.com/package/tabby-agent) installed.
    ```sh
    npm install --global tabby-agent
    ```
- **LSP Client**: The Neovim built-in LSP client, or a Vim plugin that provides an LSP client. Supported LSP clients include:
    - The Neovim built-in LSP client, with the [nvim-lspconfig](https://github.com/neovim/nvim-lspconfig) plugin installed.
    - More clients are in development.
- **Textprop Support**: Neovim, or Vim v9.0+ with `+textprop` features enabled. This is required for inline completion ghost text rendering.

## Installation

You can install the Tabby plugin using your favorite plugin manager by simply adding `TabbyML/vim-tabby` to the registry.  

Here is a detailed example setup with advanced options, based on [Neovim](https://neovim.io/), [Lazy.nvim](https://github.com/folke/lazy.nvim), and [nvim-lspconfig](https://github.com/neovim/nvim-lspconfig).

```lua
-- ~/.config/nvim/init.lua
require("lazy").setup({
  -- other plugins
  -- ...
  -- Tabby plugin
  { 
    "TabbyML/vim-tabby",
    lazy = false,
    dependencies = {
      "neovim/nvim-lspconfig",
    },
    init = function()
      vim.g.tabby_agent_start_command = {"npx", "tabby-agent", "--stdio"}
      vim.g.tabby_inline_completion_trigger = "auto"
    end,
  },
})
```
After setting up the plugin, you can open a file in Neovim and use `:LspInfo` to check if the Tabby plugin is successfully connected.

## Getting Started

### 1. Setup Tabby Server
The Tabby plugin requires a Tabby server to work. Follow the [documentation](https://tabby.tabbyml.com/docs/installation/) to install and [create your account](https://tabby.tabbyml.com/docs/quick-start/register-account/).

### 2. Connect to the Server
Edit the tabby-agent config file located at `~/.tabby-client/agent/config.toml` to set up the server endpoint and token. This file may have been auto-created if you have previously used the tabby-agent or Tabby plugin for other IDEs. You can also manually create this file.

    ```toml
    [server]
    endpoint = "http://localhost:8080"
    token = "your-auth-token"
    ```

### 3. Code Completion 
Tabby suggests code completions in real-time as you write code. You can also trigger the completion manually by pressing `<C-\>`. To accept suggestions, simply press the `<Tab>` key. You can also continue typing or explicitly press `<C-\>` again to dismiss it.

## Known Conflicts

- Tabby will attempt to set up the `<Tab>` key mapping to accept the inline completion and will fall back to the original function mapped to it. There could be a conflict with other plugins that also map the `<Tab>` key. In such cases, you can use a different keybinding to accept the completion to avoid conflicts.

- Tabby internally utilizes the `<C-R><C-O>` command to insert the completion. If you have mapped it to other functions, the insertion of the completion text may fail.

## Configurations

You can find a detailed explanation of tabby-agent configurations in the [Tabby online documentation](https://tabby.tabbyml.com/docs/extensions/configurations/).

Here is a table of all configuration variables that can be set when the Tabby plugin initializes:

| Variable | Default | Description |
| --- | --- | --- |
| `g:tabby_agent_start_command` | `["npx", "tabby-agent", "--stdio"]` | The command to start the tabby-agent |
| `g:tabby_inline_completion_trigger` | `"auto"` | The trigger mode of inline completion, can be `"auto"` or `"manual"` |
| `g:tabby_inline_completion_keybinding_accept` | `"<Tab>"` | The keybinding to accept the inline completion |
| `g:tabby_inline_completion_keybinding_trigger_or_dismiss` | `"<C-\>"` | The keybinding to trigger or dismiss the inline completion |
| `g:tabby_inline_completion_insertion_leading_key` | `"\<C-R>\<C-O>="` | The leading key sequence to insert the inline completion text |

## Contributing

Repository [TabbyML/vim-tabby](https://github.com/TabbyML/vim-tabby) is for releasing Tabby plugin for Vim and Neovim. If you want to contribute to Tabby plugin, please check our main repository [TabbyML/tabby](https://github.com/TabbyML/tabby/tree/main/clients/vim).

## License

[Apache-2.0](https://github.com/TabbyML/tabby/blob/main/LICENSE)
