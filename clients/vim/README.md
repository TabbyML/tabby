# Tabby VIM extension

## Requirements

1. VIM 9.0+ with `+job` and `+textprop` features enabled. NeoVIM is not supported at the moment.
2. Node.js 16.0+.

## Getting started

### Plug
```
; Make sure that the filetype plugin has been enabled.
filetype plugin on

Plug 'TabbyML/tabby', {'rtp': 'clients/vim'}
let g:tabby_server_url = 'http://127.0.0.1:5000'
```

## Usage

1. In insert mode, Tabby will show code suggestion when you stop typing. Press `<Tab>` to accpet the current suggestion, `<M-]>` to see the next suggestion, `<M-[>` to see previous suggestion, or `<C-]>` to dismiss.

2. Use command `:Tabby enable` to enable, `:Tabby disable` to disable Tabby, and `:Tabby status` to check status.
