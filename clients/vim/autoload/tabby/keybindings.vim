" Keybindings for Tabby

if exists('g:autoloaded_tabby_keybindings')
  finish
endif
let g:autoloaded_tabby_keybindings = 1

function! tabby#keybindings#Map()
  " map `tabby#Accept`
  if toupper(g:tabby_keybinding_accept) == '<TAB>'
    " to solve <Tab> binding conflicts, we store the original <Tab> mapping and fallback to it when tabby completions is not shown
    if !empty(mapcheck('<Tab>', 'i'))
      " fallback to the original <Tab> mapping
      let tab_maparg = maparg('<Tab>', 'i', 0, 1)
      if toupper(tab_maparg.rhs) == '<NOP>'
        " if the original <Tab> mapping is <nop>, no need to fallback
        imap <script><silent><nowait><expr> <Tab> tabby#Accept()
      else
        " warp as function if rhs is expr, otherwise encode rhs as json
        let fallback_rhs = tab_maparg.expr ? '{ -> ' . tab_maparg.rhs . ' }' : substitute(json_encode(tab_maparg.rhs), '<', '\\<', 'g')
        " inject <SID>
        let fallback_rhs = substitute(fallback_rhs, '<SID>', "\<SNR>" . get(tab_maparg, 'sid') . '_', 'g')
        exec 'imap ' . (get(tab_maparg, 'script') ? '<script>' : '') . '<silent><nowait><expr> <Tab> tabby#Accept(' . fallback_rhs . ')'
      endif
    else
      " fallback to input \t
      imap <script><silent><nowait><expr> <Tab> tabby#Accept("\t")
    endif
  else
    " map directly without fallback if the user has set keybinding to other than <Tab>
    exec 'imap <script><silent><nowait><expr> ' . g:tabby_keybinding_accept . ' tabby#Accept()'
  endif

  " map `tabby#TriggerOrDismiss`, default to <C-\>
  exec 'imap <script><silent><nowait><expr> ' . g:tabby_keybinding_trigger_or_dismiss . ' tabby#TriggerOrDismiss()'
endfunction
