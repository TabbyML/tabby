# Tabby VIM extension

Tabby is compatible with both Vim and NeoVim text editor via a plugin.

## Requirements

Before installing the plugin you will need to have installed:

1. VIM 9.0+ with `+job` and `+textprop` features enabled, or NeoVIM 0.6.0+.
2. Node.js 16.0+.

## Getting started

You can either install TabbyML vim extension using [Vim-Plug](https://github.com/junegunn/vim-plug), [Packer](https://github.com/wbthomason/packer.nvim) or [Lazy](https://github.com/folke/lazy.nvim).

### 🔌 Vim-Plug

[Vim-Plug](https://github.com/junegunn/vim-plug) is a minimalist Vim plugin manager that you can use to install TabbyML plugin.
You can install Vim-Plug by following these [intructions](https://github.com/junegunn/vim-plug#installation).



You will need to edit your vim config file (`~/.vimrc` for vim and `~/.config/nvim/init.vim` for neovim) and copy paste the following lines in it (between the `plug#begin` and `plug#end` lines)


```
" Make sure that the filetype plugin has been enabled.
filetype plugin on

" Add this to the vim-plug config
Plug 'TabbyML/tabby', {'rtp': 'clients/vim'}

" Set URL of Tabby server
let g:tabby_server_url = 'http://127.0.0.1:8080'
```

Note that you can change the tabby server url here.


You then need to actually install the plugin, to do so you need to type in your vim command.

```
:PlugInstall
```
You should see the tabbyML plugin beeing installed.


### 📦 Packer and Lazy
You first need to install either [Packer](https://github.com/wbthomason/packer.nvim) or [Lazy](https://github.com/folke/lazy.nvim).

In this case, you first need to clone the repo in your machine
```
git clone https://github.com/TabbyML/tabby.git ~/tabby
```
You will need to edit `~/.config/nvim/init.vim` for and copy paste the following lines in it.

```
" For lazy
return { name = "tabby", dir = '~/tabby/clients/vim', enabled = true }

" For packer
use {'~/tabby/clients/vim', as = 'tabby', enabled = true}

" Set URL of Tabby server

" With Lua
vim.g.tabby_server_url = 'http://127.0.0.1:8080'

" With VimScript
let g:tabby_server_url = 'http://127.0.0.1:8080'
```
> In the future, the ideal would be to export the Vim extension to a separate Git repository. This would simplify the installation process [#252](https://github.com/TabbyML/tabby/issues/252).

## Checking the installation

Once the plugin is installed you can check if the install was done sucessfully by doing in your vim command

```
:Tabby status
```

You should see
```
Tabby is online
```

If you se `Tabby cannot connect to the server` it means that you need to start the tabby server first. Refer to this [documentation](https://tabby.tabbyml.com/docs/installation/)

## Usage

1. In insert mode, Tabby will show code suggestion when you stop typing. Press `<Tab>` to accpet the current suggestion, `<M-]>` to see the next suggestion, `<M-[>` to see previous suggestion, or `<C-]>` to dismiss.
2. Use command `:Tabby enable` to enable, `:Tabby disable` to disable Tabby, and `:Tabby status` to check status.
3. Use command `:help Tabby` for more information.

## Configuration

### KeyBindings

The default key bindings for accept/dismiss(`<Tab>/<C-]>`) can be customized
with the following global settings.

```vimscript
let g:tabby_accept_binding = '<Tab>'
let g:tabby_dismiss_binding = '<C-]>'
```
