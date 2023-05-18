if exists('g:loaded_tabby')
  finish
endif
let g:loaded_tabby = 1

call tabby#Start()

command! -nargs=* -complete=customlist,tabby#CompleteCommands Tabby call tabby#Command(<q-args>)

imap  <script><silent><nowait><expr>  <Tab>  tabby#Accept(pumvisible() ? "\<C-N>" : "\t")
imap  <script><silent><nowait><expr>  <C-]>  tabby#Dismiss("\<C-]>")

imap  <Plug>(tabby-next)  <Cmd>call tabby#Next()<CR>
imap  <Plug>(tabby-prev)  <Cmd>call tabby#Prev()<CR>
if empty(mapcheck('<M-]>', 'i'))
  imap <M-]> <Plug>(tabby-next)
endif
if empty(mapcheck('<M-[>', 'i'))
  imap <M-[> <Plug>(tabby-prev)
endif

augroup tabby
  autocmd!
  autocmd CursorMovedI  *  call tabby#Clear()
  autocmd TextChangedI  *  call tabby#Schedule()
  autocmd BufLeave      *  call tabby#Clear()
  autocmd InsertLeave   *  call tabby#Clear()
augroup END

silent! execute 'helptags' fnameescape(expand('<sfile>:h:h') . '/doc')
