if exists('g:autoloaded_tabby')
  finish
endif
let g:autoloaded_tabby = 1

let g:tabby_version = "2.0.0-dev"

function! tabby#Setup()
  call tabby#lsp#Setup()
  call tabby#inline_completion#Setup()
endfunction
