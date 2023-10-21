if exists('g:loaded_tabby')
  finish
endif
let g:loaded_tabby = 1

command! -nargs=* -complete=customlist,tabby#commands#Complete Tabby call tabby#commands#Main(<q-args>)
silent! execute 'helptags' fnameescape(expand('<sfile>:h:h') . '/doc')

augroup tabby
  autocmd!
  autocmd VimEnter                      *  call tabby#OnVimEnter()
  autocmd VimLeave                      *  call tabby#OnVimLeave()
  autocmd TextChangedI,CompleteChanged  *  call tabby#OnTextChanged()
  autocmd CursorMovedI                  *  call tabby#OnCursorMoved()
  autocmd InsertLeave,BufLeave          *  call tabby#OnInsertLeave()
augroup END
