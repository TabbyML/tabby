if exists('g:autoloaded_tabby_inline_completion_utils')
  finish
endif
let g:autoloaded_tabby_inline_completion_utils = 1

function! tabby#inline_completion#utils#GetCurrentOffset()
  return line2byte(line(".")) + col(".") - 1
endfunction

function! tabby#inline_completion#utils#GetCharCountFromCol()
  return strchars(strpart(getline('.'), 0, col('.') - 1))
endfunction

function! tabby#inline_completion#utils#GetTimestamp()
  return float2nr(reltimefloat(reltime()) * 1000)
endfunction
