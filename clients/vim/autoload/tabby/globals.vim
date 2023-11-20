
" Global variables of Tabby plugin. Include options and internal variables.

if exists('g:autoloaded_tabby_globals')
  finish
endif

function! tabby#globals#Load()
  let g:autoloaded_tabby_globals = 1

  " See *Tabby-options* section in `doc/tabby.txt` for more details about options.

  " The trigger mode of compleiton, default is "auto".
  " - auto: Tabby automatically show inline completion when you stop typing.
  " - manual: You need to press <C-\> to show inline completion.
  let g:tabby_trigger_mode = get(g:, 'tabby_trigger_mode', 'auto')


  " Tabby requires Node.js version 18.0 or higher to run the tabby agent.
  " Specify the binary of Node.js, default is "node", which means search in $PATH.
  let g:tabby_node_binary = get(g:, 'tabby_node_binary', 'node')

  " The script of tabby agent.
  let g:tabby_node_script = expand('<script>:h:h:h') . '/node_scripts/tabby-agent.js'


  " Tabby use `getbufvar('%', '&filetype')` to get filetype of current buffer, and
  " then use `g:tabby_filetype_dict` to map it to language identifier.
  " From: vim filetype https://github.com/vim/vim/blob/master/runtime/filetype.vim
  " To: vscode language identifier https://code.visualstudio.com/docs/languages/identifiers#_known-language-identifiers
  " Not listed filetype will be used as language identifier directly.
  let s:default_filetype_dict = #{
    \ bash: "shellscript",
    \ sh: "shellscript",
    \ cs: "csharp",
    \ objc: "objective-c",
    \ objcpp: "objective-cpp",
    \ make: "makefile",
    \ cuda: "cuda-cpp",
    \ text: "plaintext",
    \ }
  let g:tabby_filetype_dict = get(g:, 'tabby_filetype_dict', {})
  let g:tabby_filetype_dict = extend(s:default_filetype_dict, g:tabby_filetype_dict)

  " Keybinding of accept completion, default is "<Tab>".
  let g:tabby_keybinding_accept = get(g:, 'tabby_keybinding_accept', '<Tab>')

  " Keybinding of trigger or dismiss completion, default is "<C-\>".
  let g:tabby_keybinding_trigger_or_dismiss = get(g:, 'tabby_keybinding_trigger_or_dismiss', '<C-\>')


  " Version of Tabby plugin. Not configurable.
  let g:tabby_version = "1.1.1"
endfunction