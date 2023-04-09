if exists('g:autoloaded_tabby')
  finish
endif
let g:autoloaded_tabby = 1

" Commands

let s:commands = {}

function! tabby#Command(args)
  let args = split(a:args, ' ')
  if len(args) < 1
    call tabby#Status()
    return
  endif
  if args[0] == 'enable'
    call tabby#Enable()
    call tabby#Status()
  elseif args[0] == 'disable'
    call tabby#Disable()
    call tabby#Status()
  elseif args[0] == 'server'
    if len(args) < 2
      echo 'Usage: Tabby server <url>'
      return
    endif
    call tabby#SetServerUrl(args[1])
    echo 'Tabby server URL set to ' . args[1]
  elseif args[0] == 'status'
    call tabby#Status()
  else
    echo 'Unknown command'
  endif
endfunction

" Settings

if !exists('g:tabby_enabled')
  let g:tabby_enabled = v:true
endif

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

function! tabby#SetServerUrl(url)
  let g:tabby_server_url = a:url
  call s:UpdateServerUrl()
endfunction

" Node job control

function! tabby#Start()
  if !g:tabby_enabled || tabby#Running()
    return
  endif

  if !exists('*job_start') || !exists('*prop_type_add')
    let s:errmsg = 'Tabby requires Vim 9.0+ with +job and +textprop support.'
    return
  endif

  hi def TabbyCompletion guifg=#808080 ctermfg=8
  let s:prop_type = 'TabbyCompletion'
  if prop_type_get(s:prop_type) != v:null
    call prop_type_delete(s:prop_type)
  endif
  call prop_type_add(s:prop_type, {'highlight': 'TabbyCompletion'})

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
  let s:tabby = job_start(command, #{
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
    call job_stop(s:tabby)
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
  call ch_sendexpr(s:tabby, #{
    \ func: 'setServerUrl',
    \ args: [g:tabby_server_url],
    \ })
endfunction

function! s:GetCompletion(id)
  if !tabby#Running()
    return
  endif

  let l:language = s:GetLanguage()
  if l:language == 'unknown'
    return
  endif
  call ch_sendexpr(s:tabby, #{
    \ func: 'getCompletion',
    \ args: [#{
      \ prompt: s:GetPrompt(),
      \ language: l:language,
      \ }],
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
  call ch_sendexpr(s:tabby, #{
    \ func: 'postEvent',
    \ args: [#{
      \ type: a:event_type,
      \ id: s:completion.id,
      \ index: s:completion.choices[s:completion_index].index,
      \ }],
    \ })
endfunction

function! s:HandleNotification(channel, data)
  if a:data.event == 'statusChanged'
    let s:tabby_status = a:data.status
  endif
endfunction

function! s:HandleCompletion(id, channel, data)
  if !exists('s:trigger_id') || (a:id != s:trigger_id)
    return
  endif
  if a:data == v:null
    return
  endif
  if (type(a:data.choices) == v:t_list) && (len(a:data.choices) > 0)
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
  let delay = 150
  let s:scheduled = timer_start(delay, function('tabby#Trigger'))
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
  let languages = #{
    \ javascript: 'javascript',
    \ python: 'python',
    \ }
  if has_key(languages, filetype)
    return languages[filetype]
  else
    return 'unknown'
  endif
endfunction

" Completion control

function! tabby#Show()
  call s:RemoveProp()
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
  call prop_add(line('.'), col('.'), #{
    \ type: s:prop_type,
    \ text: lines[0],
    \ })
  for line in lines[1:]
    call prop_add(line('.'), 0, #{
      \ type: s:prop_type,
      \ text: line,
      \ text_align: 'below',
      \ })
  endfor
  let s:prop_shown_lines = lines
  call s:PostEvent('view')
endfunction

function! tabby#ComsumeInsertion()
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
    let insertion = "\<C-R>\<C-O>=tabby#ComsumeInsertion()\<CR>"
  else
    let current_line = getbufline('%', line('.'), line('.'))[0]
    let suffix_chars_to_replace = len(current_line) - col('.') + 1
    let s:text_to_insert = join(lines, "\n")
    let insertion = repeat("\<Del>", suffix_chars_to_replace) . "\<C-R>\<C-O>=tabby#ComsumeInsertion()\<CR>"
  endif
  call s:RemoveProp()
  call s:PostEvent('select')
  return insertion
endfunction

function! tabby#Dismiss(fallback)
  if !exists('s:prop_shown_lines')
    return a:fallback
  endif
  call s:RemoveProp()
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
  call s:RemoveProp()
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

function! s:RemoveProp()
  call prop_remove(#{
    \ type: s:prop_type,
    \ all: v:true,
    \ })
  if exists('s:prop_shown_lines')
    unlet s:prop_shown_lines
  endif
endfunction
