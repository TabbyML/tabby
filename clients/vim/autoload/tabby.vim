if exists('g:autoloaded_tabby')
  finish
endif
let g:autoloaded_tabby = 1

" Commands

let s:commands = {}

function! s:commands.status(...)
  call tabby#Status()
endfunction

function! s:commands.enable(...)
  call tabby#Enable()
  call tabby#Status()
endfunction

function! s:commands.disable(...)
  call tabby#Disable()
  call tabby#Status()
endfunction

function! s:commands.toggle(...)
  call tabby#Toggle()
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

function! tabby#CompleteCommands(arglead, cmd, pos)
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
    return filter(candidates, { idx, val ->
      \ val[0:end_index] ==# a:arglead
      \})
  endif
endfunction

function! tabby#Command(args)
  let args = split(a:args, ' ')
  if len(args) < 1
    call tabby#Status()
    echo 'Use :help Tabby to see available commands.'
    return
  endif
  if has_key(s:commands, args[0])
    call s:commands[args[0]](args[1:])
  else
    echo 'Unknown command'
  endif
endfunction

" Settings

if !exists('g:tabby_enabled')
  let g:tabby_enabled = v:true
endif

if !exists('g:tabby_suggestion_delay')
  let g:tabby_suggestion_delay = 150
endif

if !exists('g:tabby_filetype_to_languages')
  " From: vim filetype https://github.com/vim/vim/blob/master/runtime/filetype.vim
  " To: vscode language identifier https://code.visualstudio.com/docs/languages/identifiers#_known-language-identifiers
  " Not listed filetype will be used as language identifier directly.
  let g:tabby_filetype_to_languages = {
    \ "bash": "shellscript",
    \ "cs": "csharp",
    \ "objc": "objective-c",
    \ "objcpp": "objective-cpp",
    \ }
endif

function! tabby#SetServerUrl(url)
  let g:tabby_server_url = a:url
  call s:UpdateServerUrl()
endfunction

" Node job control

function! tabby#Enable()
  let g:tabby_enabled = v:true
  if !tabby#Running()
    call tabby#Start()
  endif
endfunction

function! tabby#Disable()
  let g:tabby_enabled = v:false
  if tabby#Running()
    call tabby#Stop()
  endif
endfunction

function! tabby#Toggle()
  if g:tabby_enabled
    call tabby#Disable()
  else
    call tabby#Enable()
  endif
endfunction

function! tabby#Start()
  if !g:tabby_enabled || tabby#Running()
    return
  endif

  let check_job = tabby#job#Check()
  if !check_job.ok
    let s:errmsg = check_job.message
    return
  endif

  let check_inline_completion = tabby#inline_completion#Check()
  if !check_inline_completion.ok
    let s:errmsg = check_inline_completion.message
    return
  endif
  call tabby#inline_completion#Init()

  if !executable('node')
    let s:errmsg = 'Tabby requires node to be installed.'
    return
  endif

  let tabby_root = expand('<sfile>:h:h')
  let node_script = tabby_root . '/node_scripts/dist/tabby.js'
  if !filereadable(node_script)
    let s:errmsg = 'Tabby node script should be built first. Run `yarn && yarn build` in `./node_scripts`.'
    return
  endif

  let s:tabby_status = 'connecting'

  let command = 'node ' . node_script
  let s:tabby = tabby#job#Start(command, #{
    \ in_mode: 'json',
    \ out_mode: 'json',
    \ out_cb: function('s:HandleNotification'),
    \ exit_cb: function('s:HandleExit'),
    \ })

  if exists('g:tabby_server_url')
    call s:UpdateServerUrl()
  endif
endfunction

function! tabby#Stop()
  if tabby#Running()
    call tabby#job#Stop(s:tabby)
  endif
endfunction

function! tabby#Running()
  return exists('s:tabby')
endfunction

function! tabby#Status()
  if !g:tabby_enabled
    echo 'Tabby is disabled'
    return
  endif
  if tabby#Running()
    if s:tabby_status == 'ready'
      echo 'Tabby is online'
    elseif s:tabby_status == 'connecting'
      echo 'Tabby is connecting to server'
    elseif s:tabby_status == 'disconnected'
      echo 'Tabby cannot connect to server'
    endif
  elseif exists('s:errmsg')
    echo s:errmsg
  else
    echo 'Tabby is enabled but not running'
  endif
endfunction

function! s:UpdateServerUrl()
  if !tabby#Running()
    return
  endif
  call tabby#job#Send(s:tabby, #{
    \ func: 'setServerUrl',
    \ args: [g:tabby_server_url],
    \ })
endfunction

function! s:GetCompletion(id)
  if !tabby#Running()
    return
  endif

  call tabby#job#Send(s:tabby, #{
    \ func: 'api.default.completionsV1CompletionsPost',
    \ args: [#{
      \ prompt: s:GetPrompt(),
      \ language: s:GetLanguage(),
      \ }],
    \ cancelPendingRequest: v:true,
    \ }, #{
    \ callback: function('s:HandleCompletion', [a:id]),
    \ })
endfunction

function! s:PostEvent(event_type)
  if !tabby#Running()
    return
  endif
  if !exists('s:completion') || !exists('s:completion_index')
    return
  endif
  call tabby#job#Send(s:tabby, #{
    \ func: 'api.default.eventsV1EventsPost',
    \ args: [#{
      \ type: a:event_type,
      \ completion_id: s:completion.id,
      \ choice_index: s:completion.choices[s:completion_index].index,
      \ }],
    \ })
endfunction

function! s:HandleNotification(channel, data)
  if has_key(a:data, 'event') && (a:data.event == 'statusChanged')
    let s:tabby_status = a:data.status
  endif
endfunction

function! s:HandleCompletion(id, channel, data)
  if !exists('s:trigger_id') || (a:id != s:trigger_id)
    return
  endif
  if (type(a:data) == v:t_dict) && has_key(a:data, 'choices') &&
    \ (type(a:data.choices) == v:t_list) && (len(a:data.choices) > 0)
    let s:completion = a:data
    let s:completion_index = 0
    call tabby#Show()
  endif
endfunction

function! s:HandleExit(channel, data)
  if exists('s:tabby')
    unlet s:tabby
  endif
  if exists('s:tabby_status')
    unlet s:tabby_status
  endif
endfunction

" Completion trigger

function! tabby#Schedule()
  if !tabby#Running()
    return
  endif
  call tabby#Clear()
  let s:scheduled = timer_start(g:tabby_suggestion_delay, function('tabby#Trigger'))
endfunction

function! tabby#Trigger(timer)
  if !tabby#Running()
    return
  endif
  call tabby#Clear()
  let id = join(reltime(), '.')
  let s:trigger_id = id
  call s:GetCompletion(id)
endfunction

function! s:GetPrompt()
  let max_lines = 20
  let first_line = max([1, line('.') - max_lines])
  let lines = getbufline('%', first_line, line('.'))
  let lines[-1] = lines[-1][:col('.') - 2]
  return join(lines, "\n")
endfunction

function! s:GetLanguage()
  let filetype = getbufvar('%', '&filetype')
  if has_key(g:tabby_filetype_to_languages, filetype)
    return g:tabby_filetype_to_languages[filetype]
  else
    return filetype
  endif
endfunction

" Completion control

function! tabby#Show()
  call s:RemoveCompletion()
  if !s:CompletionAvailable()
    return
  endif
  if s:completion_index == len(s:completion.choices)
    " An empty choice after last and before first
    return
  endif
  let choice = s:completion.choices[s:completion_index]
  if (type(choice.text) != v:t_string) || (len(choice.text) == 0)
    return
  endif
  let lines = split(choice.text, "\n")
  call tabby#inline_completion#Show(lines)
  let s:prop_shown_lines = lines
  call s:PostEvent('view')
endfunction

function! tabby#ConsumeInsertion()
  if !exists('s:text_to_insert')
    return ''
  else
    let text = s:text_to_insert
    unlet s:text_to_insert
    return text
  endif
endfunction

function! tabby#Accept(fallback)
  if !exists('s:prop_shown_lines')
    return a:fallback
  endif
  let lines = s:prop_shown_lines
  if len(lines) == 1
    let s:text_to_insert = lines[0]
    let insertion = "\<C-R>\<C-O>=tabby#ConsumeInsertion()\<CR>"
  else
    let current_line = getbufline('%', line('.'), line('.'))[0]
    let suffix_chars_to_replace = len(current_line) - col('.') + 1
    let s:text_to_insert = join(lines, "\n")
    let insertion = repeat("\<Del>", suffix_chars_to_replace) . "\<C-R>\<C-O>=tabby#ConsumeInsertion()\<CR>"
  endif
  call s:RemoveCompletion()
  call s:PostEvent('select')
  return insertion
endfunction

function! tabby#Dismiss(fallback)
  if !exists('s:prop_shown_lines')
    return a:fallback
  endif
  call s:RemoveCompletion()
  return ''
endfunction

function! tabby#Next()
  if !s:CompletionAvailable()
    return
  endif
  if !exists('s:prop_shown_lines')
    if s:completion_index == len(s:completion.choices)
      let s:completion_index = 0
    endif
  else
    let s:completion_index += 1
    if s:completion_index > len(s:completion.choices)
      let s:completion_index = 0
    endif
  endif
  call tabby#Show()
endfunction

function! tabby#Prev()
  if !s:CompletionAvailable()
    return
  endif
  if !exists('s:prop_shown_lines')
    if s:completion_index == len(s:completion.choices)
      let s:completion_index = len(s:completion.choices) - 1
    endif
  else
    let s:completion_index -= 1
    if s:completion_index < 0
      let s:completion_index = len(s:completion.choices)
    endif
  endif
  call tabby#Show()
endfunction

function! tabby#Clear()
  call s:RemoveCompletion()
  if exists('s:scheduled')
    call timer_stop(s:scheduled)
    unlet s:scheduled
  endif
  if exists('s:trigger_id')
    unlet s:trigger_id
  endif
  if exists('s:completion')
    unlet s:completion
  endif
  if exists('s:completion_index')
    unlet s:completion_index
  endif
endfunction

function! s:CompletionAvailable()
  if !exists('s:completion') || !exists('s:completion_index')
    return v:false
  endif
  if (type(s:completion.choices) != v:t_list) || (len(s:completion.choices) == 0)
    return v:false
  endif
  return v:true
endfunction

function! s:RemoveCompletion()
  call tabby#inline_completion#Clear()
  if exists('s:prop_shown_lines')
    unlet s:prop_shown_lines
  endif
endfunction
