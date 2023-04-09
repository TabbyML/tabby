# How to Use This VIM Plugin

## Requirements

1. VIM 9.0+ with `+job` and `+textprop` features enabled. NeoVIM is not supported at the moment.
2. Node.js 16.0+, with `yarn` or `npm` installed.

## Setup

1. Build node scripts in `node_scripts/` directory. Use `yarn` as example. You can also use `npm` instead.
   ```bash
   cd node_scripts
   yarn && yarn build
   cd ..
   ```

2. Copy this directory to your VIM plugin directory.
   ```bash
   cp -r . ~/.vim/pack/plugins/start/tabby.vim
   ```

3. (Optional) Set Tabby server URL in your `vimrc` file. If you do not set a URL, the default value is `http://127.0.0.1:5000`.
   ```vim
   let g:tabby_server_url = 'http://127.0.0.1:5000'
   ```

4. (Optional) Turn on `filetype` plugin for better compatibility.
   ```vim
   filetype plugin on
   ```

## Usage

1. In insert mode, Tabby will show code suggestion when you stop typing. Press `<Tab>` to accpet the current suggestion, `<M-]>` to see the next suggestion, `<M-[>` to see previous suggestion, or `<C-]>` to dismiss.

2. Use command `:Tabby enable` to enable, `:Tabby disable` to disable Tabby, and `:Tabby status` to check status.
