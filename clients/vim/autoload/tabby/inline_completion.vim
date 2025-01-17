if exists('g:autoloaded_tabby_inline_completion')
  finish
endif
let g:autoloaded_tabby_inline_completion = 1

let g:tabby_inline_completion_source = get(g:, 'tabby_inline_completion_source', #{
  \ RequestInlineCompletion: { params, Callback -> tabby#lsp#RequestInlineCompletion(params, Callback) },
  \ NotifyEvent: { params -> tabby#lsp#NotifyEvent(params) },
  \ CancelRequest: { id -> tabby#lsp#CancelRequest(id) },
  \ })

function! tabby#inline_completion#Setup()
  augroup tabby_inline_completion_install
    autocmd!
    autocmd User tabby_lsp_on_buffer_attached call tabby#inline_completion#Install()
  augroup end
endfunction

function! tabby#inline_completion#Install()
  call tabby#inline_completion#events#Install()
  call tabby#inline_completion#keybindings#Setup()
  call tabby#inline_completion#virtual_text#Setup()
endfunction

function! tabby#inline_completion#Uninstall()
  call tabby#inline_completion#events#Uninstall()
endfunction
