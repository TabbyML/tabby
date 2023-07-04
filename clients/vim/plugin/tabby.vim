if exists('g:loaded_tabby')
  finish
endif
let g:loaded_tabby = 1

call tabby#Start()

command! -nargs=* -complete=customlist,tabby#CompleteCommands Tabby call tabby#Command(<q-args>)

if !exists('g:tabby_accept_binding')
  let g:tabby_accept_binding = '<Tab>'
endif
if !exists('g:tabby_dismiss_binding')
  let g:tabby_dismiss_binding = '<C-]>'
endif

function s:MapKeyBindings()
  " map `tabby#Accept`
  if g:tabby_accept_binding == '<Tab>'
    " to solve <Tab> binding conflicts, we store the original <Tab> mapping and fallback to it when tabby completions is not shown
    if exists('g:tabby_binding_tab_fallback')
      " map directly if the user has set a custom fallback method
      imap <script><silent><nowait><expr> <Tab> tabby#Accept(g:tabby_binding_tab_fallback)
    else
      if !empty(mapcheck('<Tab>', 'i'))
        " fallback to the original <Tab> mapping
        let tab_maparg = maparg('<Tab>', 'i', v:false, v:true)
        " warp as function if rhs is expr
        let fallback_rhs = tab_maparg.expr ? '{ -> ' . tab_maparg.rhs . ' }' : tab_maparg.rhs
        " inject <SID>
        let fallback_rhs = substitute(fallback_rhs, '<SID>', "\<SNR>" . get(tab_maparg, 'sid') . '_', 'g')
        exec 'imap <script><silent><nowait><expr> <Tab> tabby#Accept(' . fallback_rhs . ')'
      else
        " fallback to input \t
        imap <script><silent><nowait><expr> <Tab> tabby#Accept("\t")
      endif
    endif
  else
    " map directly without fallback if the user has set a custom binding
    exec 'imap <script><silent><nowait><expr> ' . g:tabby_accept_binding . ' tabby#Accept()'
  endif

  " map `tabby#Dismiss`
  if g:tabby_accept_binding == '<C-]>'
    imap <script><silent><nowait><expr> <C-]> tabby#Dismiss("\<C-]>")
  else
    " map directly without fallback if the user has set a custom binding
    exec 'imap <script><silent><nowait><expr> ' . g:tabby_dismiss_binding . ' tabby#Dismiss()'
  endif

  " map `tabby#Next` and `tabby#Prev`
  imap  <Plug>(tabby-next)  <Cmd>call tabby#Next()<CR>
  imap  <Plug>(tabby-prev)  <Cmd>call tabby#Prev()<CR>
  if empty(mapcheck('<M-]>', 'i'))
    imap <M-]> <Plug>(tabby-next)
  endif
  if empty(mapcheck('<M-[>', 'i'))
    imap <M-[> <Plug>(tabby-prev)
  endif
endfunction

augroup tabby
  autocmd!
  autocmd TextChangedI,CompleteChanged  *  call tabby#Schedule()
  autocmd CursorMovedI  *  call tabby#Clear()
  autocmd BufLeave      *  call tabby#Clear()
  autocmd InsertLeave   *  call tabby#Clear()
  
  " map bindings as late as possible, to avoid <Tab> binding override by other scripts
  autocmd VimEnter      *  call s:MapKeyBindings()
augroup END

silent! execute 'helptags' fnameescape(expand('<sfile>:h:h') . '/doc')
