# Tabby Plugin for Vim and NeoVim

Tabby is a self-hosted AI coding assistant that can suggest multi-line code or full functions in real-time. For more information, please check out our [website](https://tabbyml.com/) and [github](https://github.com/TabbyML/tabby).  
If you encounter any problem or have any suggestion, please [open an issue](https://github.com/TabbyML/tabby/issues/new) or join our [Slack community](https://links.tabbyml.com/join-slack) for support.

## Table of Contents

- [Tabby Plugin for Vim and NeoVim](#tabby-plugin-for-vim-and-neovim)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Installation](#installation)
    - [🔌 Vim-plug](#-vim-plug)
    - [📦 Packer.nvim](#-packernvim)
    - [💤 Lazy.nvim](#-lazynvim)
  - [Usage](#usage)
  - [Configuration](#configuration)
    - [Tabby Server](#tabby-server)
    - [Node.js Binary Path](#nodejs-binary-path)
    - [Completion Trigger Mode](#completion-trigger-mode)
    - [KeyBindings](#keybindings)
  - [Contributing](#contributing)
  - [License](#license)

## Requirements

Tabby plugin requires the following dependencies:

- Vim 9.0+ with `+job` and `+textprop` features enabled, or NeoVim 0.6.0+.
- Tabby server. You can install Tabby server locally or have it hosted on a remote server. For Tabby server installation, please refer to this [documentation](https://tabby.tabbyml.com/docs/installation/).
- [Node.js](https://nodejs.org/en/download/) version v18.0+.
  - If you need have multiple Node.js versions installed, you can use Node.js version manager such as [nvm](https://github.com/nvm-sh/nvm).
- Vim filetype plugin enabled. You can add following lines in vim config file (`~/.vimrc`). For NeoVim, filetype plugin is enabled by default, you don't need to add these lines.

  ```vim
  filetype plugin on
  ```

## Installation

You can install Tabby plugin using your favorite plugin manager. Here are some examples using popular plugin managers, you can choose one to follow.

### 🔌 [Vim-plug](https://github.com/junegunn/vim-plug)

```vim
" Example ~/.vimrc configuration
filetype plugin on

" Section for plugins managed by vim-plug
plug#begin()

" Tabby plugin
Plug 'TabbyML/vim-tabby'
" Add config here. Example config:
let g:tabby_keybinding_accept = '<Tab>'

plug#end()
```

### 📦 [Packer.nvim](https://github.com/wbthomason/packer.nvim)

```lua
--- Example Packer plugin specification

--- Add config here. Example config:
vim.g.tabby_keybinding_accept = '<Tab>'

return require('packer').startup(function(use)
  --- Tabby plugin
  use { 'TabbyML/vim-tabby' }
end)
```

### 💤 [Lazy.nvim](https://github.com/folke/lazy.nvim)

```lua
--- Example Lazy plugin specification

--- Add config here. Example config:
vim.g.tabby_keybinding_accept = '<Tab>'

return {
  { 'TabbyML/vim-tabby' }
}
```

## Usage

After installation, please exit and restart Vim or NeoVim. Then you can check the Tabby plugin status by running `:Tabby` in your vim command line. If you see any message reported by Tabby, it means the plugin is installed successfully. If you see `Not an editor command: Tabby` or any other error message, please check the installation steps.

In insert mode, Tabby plugin will show inline completion automatically when you stop typing. You can simply press `<Tab>` to accept the completion. If you want to dismiss the completion manually, you can press `<C-\>` to dismiss, and press `<C-\>` again to show the completion again.

## Configuration

### Tabby Server

You need to start the Tabby server before using the plugin. For Tabby server installation, please refer to this [documentation](https://tabby.tabbyml.com/docs/installation/).

If your Tabby server endpoint is different from the default `http://localhost:8080`, please set the endpoint in `~/.tabby-client/agent/config.toml`.

If your Tabby server requires an authentication token, remember to set it here.

```toml
# Server
# You can set the server endpoint here.
[server]
endpoint = "http://localhost:8080" # http or https URL
token = "your-auth-token"
```

### Node.js Binary Path

Normally, this config is not required as the Tabby plugin will try to find the Node.js binary in your `PATH` environment variable.  
But if you have installed Node.js in a non-standard location, or you are using a Node.js version manager such as nvm, you can set the Node.js binary path in your vim config file (`~/.vimrc` for Vim and `~/.config/nvim/init.vim` or `~/.config/nvim/init.lua` for NeoVim).

```vim
let g:tabby_node_binary = '/path/to/node'
```

```lua
--- lua
vim.g.tabby_node_binary = '/path/to/node'
```

### Completion Trigger Mode

Completion trigger mode is set to `auto` by default, Tabby plugin will show inline completion automatically when you stop typing.  
If you prefer to trigger code completion manually, add this config in your vim config file. Tabby plugin will not show inline completion automatically, you can trigger the completion manually by pressing `<C-\>`.

```vim
let g:tabby_trigger_mode = 'manual'
```

```lua
--- lua
vim.g.tabby_trigger_mode = 'manual'
```

### KeyBindings

The default key bindings for accept completion(`<Tab>`), manual trigger/dismiss(`<C-\>`) can be customized with the following global settings.

```vim
let g:tabby_keybinding_accept = '<Tab>'
let g:tabby_keybinding_trigger_or_dismiss = '<C-\>'
```

```lua
--- lua
vim.g.tabby_keybinding_accept = '<Tab>'
vim.g.tabby_keybinding_trigger_or_dismiss = '<C-\\>'
```

## Known Conflicts

- For the default settings, Tabby will attempt to set up the `<Tab>` key mapping. If Tabby's inline completion is not displayed, it will fall back to the original mapping. However, this approach might not work when there is a conflict with other plugins that also map the `<Tab>` key, as they could overwrite Tabby's mapping. In such cases, you can use a different keybinding to accept the completion and avoid conflicts.

- Tabby internally utilizes the `<C-R><C-O>` command to insert the completion. If you have mapped `<C-R>` to other functions, you won't be able to accept the completion. In such scenarios, you may need to manually modify the function `tabby#Accept()` in [`autoload/tabby.vim`](https://github.com/TabbyML/tabby/tree/main/clients/vim/autoload/tabby.vim).

## Contributing

Repository [TabbyML/vim-tabby](https://github.com/TabbyML/vim-tabby) is for releasing Tabby plugin for Vim and NeoVim. If you want to contribute to Tabby plugin, please check our main repository [TabbyML/tabby](https://github.com/TabbyML/tabby/tree/main/clients/vim).

## License

[Apache-2.0](https://github.com/TabbyML/tabby/blob/main/LICENSE)
