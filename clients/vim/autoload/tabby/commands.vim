" Commands for Tabby

if exists('g:autoloaded_tabby_commands')
  finish
endif
let g:autoloaded_tabby_commands = 1


" See `*Tabby-commands*` section in `doc/tabby.txt` for more details.

"   A dictionary contains all commands. Use name as key and function as value.
let s:commands = {}

function! s:commands.status(...)
  call tabby#Status()
endfunction

function! s:commands.version(...)
  echo g:tabby_version
endfunction

function! s:commands.help(...)
  let args = get(a:, 1, [])
  if len(args) < 1
    execute 'help Tabby'
    return
  endif
  try
    execute 'help Tabby-' . join(args, '-')
    return
  catch
  endtry
  try
    execute 'help tabby_' . join(args, '_')
    return
  catch
  endtry
  execute 'help Tabby'
endfunction

function! tabby#commands#Main(args)
  let args = split(a:args, ' ')
  if len(args) < 1
    call tabby#Status()
    echo 'Use `:help Tabby` to see available commands.'
    return
  endif
  if has_key(s:commands, args[0])
    call s:commands[args[0]](args[1:])
  else
    echo 'Unknown command.'
    echo 'Use `:help Tabby` to see available commands.'
  endif
endfunction

function! tabby#commands#Complete(arglead, cmd, pos)
  let words = split(a:cmd[0:a:pos].'#', ' ')
  if len(words) > 3
    return []
  endif
  if len(words) == 3
    if words[1] == 'help'
      let candidates = ['compatibility', 'commands', 'options', 'keybindings']
    else
      return []
    endif
  else
    let candidates = keys(s:commands)
  endif

  let end_index = len(a:arglead) - 1
  if end_index < 0
    return candidates
  else
    return filter(candidates, { idx, val -> val[0:end_index] == a:arglead })
  endif
endfunction
