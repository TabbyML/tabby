if exists('g:autoloaded_tabby_inline_completion_events')
  finish
endif
let g:autoloaded_tabby_inline_completion_events = 1

function! tabby#inline_completion#events#Install()
  augroup tabby_inline_completion_events
    autocmd!
    autocmd TextChangedI,CompleteChanged  *  call tabby#inline_completion#events#OnTextChanged()
    autocmd CursorMovedI                  *  call tabby#inline_completion#events#OnCursorMoved()
    autocmd InsertLeave,BufLeave          *  call tabby#inline_completion#events#OnInsertLeave()
  augroup END
endfunction

function! tabby#inline_completion#events#Uninstall()
  augroup tabby_inline_completion_events
    autocmd!
  augroup END
endfunction

function! tabby#inline_completion#events#OnTextChanged()
  call tabby#inline_completion#service#Clear()
  call tabby#inline_completion#service#Trigger(v:false)
endfunction

function! tabby#inline_completion#events#OnCursorMoved()
  call tabby#inline_completion#service#ClearCurrentIfNotMatch()
endfunction

function! tabby#inline_completion#events#OnInsertLeave()
  call tabby#inline_completion#service#Clear()
endfunction
