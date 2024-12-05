if exists('g:autoloaded_tabby_inline_completion_keybindings')
  finish
endif
let g:autoloaded_tabby_inline_completion_keybindings = 1

let g:tabby_inline_completion_keybinding_accept = get(g:, 'tabby_inline_completion_keybinding_accept', '<Tab>')
let g:tabby_inline_completion_keybinding_trigger_or_dismiss = get(g:, 'tabby_inline_completion_keybinding_trigger_or_dismiss', '<C-\>')

function! tabby#inline_completion#keybindings#Setup()
  " map `tabby#inline_completion#service#Accept`
  if toupper(g:tabby_inline_completion_keybinding_accept) == '<TAB>'
    " to solve <Tab> binding conflicts, we store the original <Tab> mapping and fallback to it when tabby completions is not shown
    if !empty(mapcheck('<Tab>', 'i'))
      " fallback to the original <Tab> mapping
      let tab_maparg = maparg('<Tab>', 'i', 0, 1)
      let tab_maparg_rhs = get(tab_maparg, 'rhs', '')
      if empty(tab_maparg_rhs) || toupper(tab_maparg_rhs) == '<NOP>'
        " if the original <Tab> mapping is <nop>, no need to fallback
        imap <buffer><script><silent><nowait><expr> <Tab> tabby#inline_completion#service#Accept()
      else
        " warp as function if rhs is expr, otherwise encode rhs as json
        let fallback_rhs = get(tab_maparg, 'expr') ? '{ -> ' . tab_maparg_rhs . ' }' : substitute(json_encode(tab_maparg_rhs), '<', '\\<', 'g')
        " inject <SID>
        let fallback_rhs = substitute(fallback_rhs, '<SID>', "\<SNR>" . get(tab_maparg, 'sid') . '_', 'g')
        exec 'imap <buffer>' . (get(tab_maparg, 'script') ? '<script>' : '') . '<silent><nowait><expr> <Tab> tabby#inline_completion#service#Accept(' . fallback_rhs . ')'
      endif
    else
      " fallback to input \t
      imap <buffer><script><silent><nowait><expr> <Tab> tabby#inline_completion#service#Accept("\t")
    endif
  else
    if !empty(g:tabby_inline_completion_keybinding_accept)
      " map directly without fallback if the user has set keybinding to other than <Tab>
      exec 'imap <buffer><script><silent><nowait><expr> ' . g:tabby_inline_completion_keybinding_accept . ' tabby#inline_completion#service#Accept()'
    endif
  endif

  if !empty(g:tabby_inline_completion_keybinding_trigger_or_dismiss)
    " map `tabby#inline_completion#service#TriggerOrDismiss`, default to <C-\>
    exec 'imap <buffer><script><silent><nowait><expr> ' . g:tabby_inline_completion_keybinding_trigger_or_dismiss . ' tabby#inline_completion#service#TriggerOrDismiss()'
  endif
endfunction

