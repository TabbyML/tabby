# Tabby VIM extension

## Requirements

1. VIM 9.0+ with `+job` and `+textprop` features enabled, or NeoVIM 0.6.0+.
2. Node.js 16.0+.

## Getting started

### 🔌 Vim-Plug
```
" Make sure that the filetype plugin has been enabled.
filetype plugin on

" Add this to the vim-plug config
Plug 'TabbyML/tabby', {'rtp': 'clients/vim'}

" Set URL of Tabby server
let g:tabby_server_url = 'http://127.0.0.1:8080'
```
### 📦 Packer and Lazy
In this case, you first need to clone the repo in your machine
```
git clone https://github.com/TabbyML/tabby.git ~/tabby
```
Then on the config file:
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

## Usage

1. In insert mode, Tabby will show code suggestion when you stop typing. Press `<Tab>` to accpet the current suggestion, `<M-]>` to see the next suggestion, `<M-[>` to see previous suggestion, or `<C-]>` to dismiss.
2. Use command `:Tabby enable` to enable, `:Tabby disable` to disable Tabby, and `:Tabby status` to check status.
3. Use command `:help Tabby` for more information.
