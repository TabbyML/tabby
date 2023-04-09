# How to Use This VIM Plugin

## Requirements

1. VIM 9.0+ with `+job` and `+textprop` features enabled. NeoVIM is not supported at the moment.
2. Node.js 16.0+, with `yarn` or `npm` installed.

## Getting started

* Vim Plug
```vimscript
Plug 'TabbyML/tabby', {'rtp': 'clients/vim'}

; Changed to your tabby server
let g:tabby_server_url = 'http://127.0.0.1:5000'

; Make sure you turn on file type plugin.
filetype plugin on
```

## Usage

1. In insert mode, Tabby will show code suggestion when you stop typing. Press `<Tab>` to accpet the current suggestion, `<M-]>` to see the next suggestion, `<M-[>` to see previous suggestion, or `<C-]>` to dismiss.

2. Use command `:Tabby enable` to enable, `:Tabby disable` to disable Tabby, and `:Tabby status` to check status.
