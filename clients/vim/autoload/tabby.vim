if exists('g:autoloaded_tabby')
  finish
endif
let g:autoloaded_tabby = 1

" Table of Contents
" 1. Commands: Implement *:Tabby* commands
" 2. Settings: Handle global *Tabby-options*
" 3. Node Job: Manage node process, implement IO callbacks
" 4. Scheduler: Schedule completion requests
" 5. Completion UI: Show up completion, handle hotkeys
" 6. Utils: Utility functions

" 1. Commands
" See *:Tabby* in help document for more details.
"
" Notable script-local variables:
" - s:commmands
"   A dictionary contains all commands. Use name as key and function as value.
"

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

" 2. Settings
" See *Tabby-options* in help document for more details.
"
" Available global options:
" - g:tabby_enabled
" - g:tabby_suggestion_delay
" - g:tabby_filetype_to_languages
" - g:tabby_server_url
"

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

" 3. Node Job
"
" Notable script-local variables:
" - s:tabby
"   Stores the job id of current node process
"
" - s:tabby_status
"   Syncs with status of node agent, updated by notification from agent
"
" - s:errmsg
"   Stores error message if self check failed before starting node process
"

function! tabby#Enable()
  let g:tabby_enabled = v:true
  if !tabby#IsRunning()
    call tabby#Start()
  endif
endfunction

function! tabby#Disable()
  let g:tabby_enabled = v:false
  if tabby#IsRunning()
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
  if !g:tabby_enabled || tabby#IsRunning()
    return
  endif

  let check_job = tabby#job#Check()
  if !check_job.ok
    let s:errmsg = check_job.message
    return
  endif

  let check_virtual_text = tabby#virtual_text#Check()
  if !check_virtual_text.ok
    let s:errmsg = check_virtual_text.message
    return
  endif
  call tabby#virtual_text#Init()

  if !executable('node')
    let s:errmsg = 'Tabby requires node to be installed.'
    return
  endif

  let tabby_root = expand('<script>:h:h')
  let node_script = tabby_root . '/node_scripts/tabby-agent.js'
  if !filereadable(node_script)
    let s:errmsg = 'Tabby node script should be download first. Try to run `yarn upgrade-agent`.'
    return
  endif

  let s:tabby_status = 'connecting'

  let command = 'node ' . node_script
  let s:tabby = tabby#job#Start(command, #{
    \ in_mode: 'json',
    \ out_mode: 'json',
    \ out_cb: function('s:HandleNotification'),
    \ err_cb: function('s:HandleError'),
    \ exit_cb: function('s:HandleExit'),
    \ })

  if exists('g:tabby_server_url')
    call s:UpdateServerUrl()
  endif
endfunction

function! tabby#Stop()
  if tabby#IsRunning()
    call tabby#job#Stop(s:tabby)
  endif
endfunction

function! tabby#IsRunning()
  return exists('s:tabby')
endfunction

function! tabby#Status()
  if !g:tabby_enabled
    echo 'Tabby is disabled'
    return
  endif
  if tabby#IsRunning()
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
  if !tabby#IsRunning()
    return
  endif
  call tabby#job#Send(s:tabby, #{
    \ func: 'setServerUrl',
    \ args: [g:tabby_server_url],
    \ })
endfunction

function! s:GetCompletion(id)
  if !tabby#IsRunning()
    return
  endif

  if exists('s:pending_request_id')
    call tabby#job#Send(s:tabby, #{
      \ func: 'cancelRequest',
      \ args: [s:pending_request_id],
      \ })
  endif

  let s:pending_request_id = tabby#job#Send(s:tabby, #{
    \ func: 'getCompletions',
    \ args: [#{
      \ prompt: s:GetPrompt(),
      \ language: s:GetLanguage(),
      \ }],
    \ }, #{
    \ callback: function('s:HandleCompletion', [a:id]),
    \ })
endfunction

function! s:PostEvent(event_type)
  if !tabby#IsRunning()
    return
  endif
  if !exists('s:completion') || !exists('s:choice_index')
    return
  endif
  call tabby#job#Send(s:tabby, #{
    \ func: 'postEvent',
    \ args: [#{
      \ type: a:event_type,
      \ completion_id: s:completion.id,
      \ choice_index: s:completion.choices[s:choice_index].index,
      \ }],
    \ })
endfunction

function! s:HandleNotification(channel, data)
  if (type(a:data) == v:t_dict) && has_key(a:data, 'event') && (a:data.event == 'statusChanged')
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
    let s:choice_index = 0
    call tabby#Show()
  endif
endfunction

function! s:HandleError(channel, data)
  " For Debug
  " echoerr "HandleError: " . string(a:data)
endfunction

function! s:HandleExit(channel, data)
  if exists('s:tabby')
    unlet s:tabby
  endif
  if exists('s:tabby_status')
    unlet s:tabby_status
  endif
endfunction

" 4. Scheduler
"
" Notable script-local variables:
" - s:scheduled:
"   Stores the timer id of current scheduled next trigger.
"
" - s:trigger_id:
"   Use a timestamp to identify current triggered completion request. This is
"   used to filter out outdated completion results.
"

function! tabby#Schedule()
  if !tabby#IsRunning()
    return
  endif
  call tabby#Clear()
  let s:scheduled = timer_start(g:tabby_suggestion_delay, function('tabby#Trigger'))
endfunction

function! tabby#Trigger(timer)
  if !tabby#IsRunning()
    return
  endif
  call tabby#Clear()
  let id = join(reltime(), '.')
  let s:trigger_id = id
  call s:GetCompletion(id)
endfunction


" 5. Completion UI
"
" Notable script-local variables:
" - s:completion:
"   Stores current completion data, a dictionary that has same struct as server
"   returned completion response.
"
" - s:choice_index:
"   Stores index of current choice to display. A 0-based index of choice item
"   of `s:completion.choices` array, may not equals to the value of `index`
"   field `s:completion.choices[s:choice_index].index`.
"   A exception is that when `s:choice_index` is equal to the length of
"   `s:completion.choices`. In this state, the completion UI should show nothing
"   to notice users that they are cycling from last choice forward to first
"   choice, or from first choice back to last choice.
"   This variable does not change when user dismisses the completion UI.
"
" - s:shown_lines:
"   Stores the text that are shown in completion UI. `s:shown_lines` exists or
"   not means whether the completion UI is shown or not.
"
" - s:text_to_insert:
"   Used as a buffer to store the text that should be inserted when user accepts
"   the completion. We hide completion UI first and clear `s:shown_lines` at
"   same time, then insert the text, so that we need to store the text in a
"   buffer until text is inserted.


function! tabby#Show()
  call s:HideCompletion()
  if !s:IsCompletionAvailable()
    return
  endif
  if s:choice_index == len(s:completion.choices)
    " Show empty to indicate that user is cycling back to first choice.
    return
  endif
  let choice = s:completion.choices[s:choice_index]
  if (type(choice.text) != v:t_string) || (len(choice.text) == 0)
    return
  endif
  let lines = split(choice.text, "\n")
  " split will not give an empty line if text starts with "\n" or ends with "\n"
  if choice.text[0] == "\n"
    call insert(lines, '')
  endif
  if choice.text[-1] == "\n"
    call add(lines, '')
  endif
  call tabby#virtual_text#Show(lines)
  let s:shown_lines = lines
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

" This function is designed to replace <Tab> key input, so we need a fallback
" when completion UI is not shown.
function! tabby#Accept(fallback)
  if !exists('s:shown_lines')
    return a:fallback
  endif
  let lines = s:shown_lines
  if len(lines) == 1
    let s:text_to_insert = lines[0]
    let insertion = "\<C-R>\<C-O>=tabby#ConsumeInsertion()\<CR>"
  else
    let current_line = getbufline('%', line('.'), line('.'))[0]
    let suffix_chars_to_replace = len(current_line) - col('.') + 1
    let s:text_to_insert = join(lines, "\n")
    let insertion = repeat("\<Del>", suffix_chars_to_replace) . "\<C-R>\<C-O>=tabby#ConsumeInsertion()\<CR>"
  endif
  call s:HideCompletion()
  call s:PostEvent('select')
  return insertion
endfunction

" This function is designed to replace <C-]> key input, so we need a fallback
" when completion UI is not shown.
function! tabby#Dismiss(fallback)
  if !exists('s:shown_lines')
    return a:fallback
  endif
  call s:HideCompletion()
  return ''
endfunction

function! tabby#Next()
  if !s:IsCompletionAvailable()
    return
  endif
  if !exists('s:shown_lines')
    if s:choice_index == len(s:completion.choices)
      let s:choice_index = 0
    endif
  else
    let s:choice_index += 1
    if s:choice_index > len(s:completion.choices)
      let s:choice_index = 0
    endif
  endif
  call tabby#Show()
endfunction

function! tabby#Prev()
  if !s:IsCompletionAvailable()
    return
  endif
  if !exists('s:shown_lines')
    if s:choice_index == len(s:completion.choices)
      let s:choice_index = len(s:completion.choices) - 1
    endif
  else
    let s:choice_index -= 1
    if s:choice_index < 0
      let s:choice_index = len(s:completion.choices)
    endif
  endif
  call tabby#Show()
endfunction

function! tabby#Clear()
  call s:HideCompletion()
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
  if exists('s:choice_index')
    unlet s:choice_index
  endif
endfunction

function! s:IsCompletionAvailable()
  if !exists('s:completion') || !exists('s:choice_index')
    return v:false
  endif
  if (type(s:completion.choices) != v:t_list) || (len(s:completion.choices) == 0)
    return v:false
  endif
  return v:true
endfunction

function! s:HideCompletion()
  call tabby#virtual_text#Clear()
  if exists('s:shown_lines')
    unlet s:shown_lines
  endif
endfunction

" 6. Utils

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
