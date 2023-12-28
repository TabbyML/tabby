" Main functions

if exists('g:autoloaded_tabby')
  finish
endif
let g:autoloaded_tabby = 1

let s:status = "initializing"
let s:message = ""

function! tabby#Status()
  if s:status == "initializing"
    echo 'Tabby is initializing.'
  elseif s:status == "initialization_failed"
    echo 'Tabby initialization failed.'
    echo s:message
  elseif s:status == "initialization_done"
    let agent_status = tabby#agent#Status()
    if agent_status == 'notInitialized'
      echo 'Tabby is initializing.'
    elseif agent_status == 'exited'
      echo 'Tabby agent exited unexpectedly.'
    elseif agent_status == 'ready'
      echo 'Tabby is online.'
      let agent_issues = tabby#agent#Issues()
      if len(agent_issues) > 0
        if agent_issues[0] == 'slowCompletionResponseTime'
          echo 'Completion requests appear to take too much time.'
        elseif agent_issues[0] == 'highCompletionTimeoutRate'
          echo 'Most completion requests timed out.'
        endif
      elseif g:tabby_trigger_mode == 'manual'
        echo 'You can use ' . g:tabby_keybinding_trigger_or_dismiss . 
          \ ' in insert mode to trigger completion manually.'
      elseif g:tabby_trigger_mode == 'auto'
        echo 'Automatic inline completion is enabled.'
      endif
    elseif agent_status == 'disconnected'
      echo 'Tabby cannot connect to server. Please check your settings.'
    elseif agent_status == 'unauthorized'
      echo 'Authorization required. Please set your personal token in settings.'
    endif
  endif
endfunction

function! tabby#OnVimEnter()
  call tabby#globals#Load()

  let check_job = tabby#job#Check()
  if !check_job.ok
    let s:status = "initialization_failed"
    let s:message = check_job.message
    return
  endif

  let check_virtual_text = tabby#virtual_text#Check()
  if !check_virtual_text.ok
    let s:status = "initialization_failed"
    let s:errmsg = check_virtual_text.message
    return
  endif
  call tabby#virtual_text#Init()

  let node_binary = expand(g:tabby_node_binary)
  if !executable(node_binary)
    let s:status = "initialization_failed"
    let s:message = 'Node.js binary not found. Please install Node.js version >= 18.0.'
    return
  endif

  let node_version_command = node_binary . ' --version'
  let version_output = system(node_version_command)
  let node_version = matchstr(version_output, '\d\+\.\d\+\.\d\+')
  let major_version = str2nr(split(node_version, '\.')[0])
  if major_version < 18
    let s:status = "initialization_failed"
    let s:message = 'Node.js version is too old: ' . node_version . '. Please install Node.js version >= 18.0.'
    return
  endif

  if !filereadable(g:tabby_node_script)
    let s:status = "initialization_failed"
    let s:message = 'Tabby agent script not found. Please reinstall Tabby plugin.'
    return
  endif

  let command = node_binary . ' --dns-result-order=ipv4first ' . g:tabby_node_script
  call tabby#agent#Open(command)

  call tabby#keybindings#Map()

  let s:status = "initialization_done"
endfunction

function! tabby#OnVimLeave()
  call tabby#agent#Close()
endfunction

function! tabby#OnTextChanged()
  if s:status != "initialization_done"
    return
  endif
  if g:tabby_trigger_mode == 'auto'
    " FIXME: Do not dismiss when type over the same as the completion text, or backspace in replace range.
    call tabby#Dismiss()
    call tabby#Trigger(v:false)
  endif
endfunction

function! tabby#OnCursorMoved()
  if s:current_completion_request == s:GetCompletionContext(v:false)
    return
  endif
  call tabby#Dismiss()
  if s:ongoing_request_id != 0
    call tabby#agent#CancelRequest(s:ongoing_request_id)
  endif
endfunction

function! tabby#OnInsertLeave()
  call tabby#Dismiss()
  if s:ongoing_request_id != 0
    call tabby#agent#CancelRequest(s:ongoing_request_id)
  endif
endfunction

function! tabby#TriggerOrDismiss()
  if s:status != "initialization_done"
    return ''
  endif
  if s:current_completion_response != {}
    call tabby#Dismiss()
  else
    call tabby#Trigger(v:true)
  endif
  return ''
endfunction

" Store the context of ongoing completion request
let s:current_completion_request = {}
let s:ongoing_request_id = 0

function! tabby#Trigger(is_manual)
  if s:status != "initialization_done"
    return
  endif
  if s:ongoing_request_id != 0
    call tabby#agent#CancelRequest(s:ongoing_request_id)
  endif
  let s:current_completion_request = s:GetCompletionContext(a:is_manual)
  let request = s:current_completion_request
  let OnResponse = { response -> s:HandleCompletionResponse(request, response) }
  let s:ongoing_request_id = tabby#agent#ProvideCompletions(request, OnResponse)
endfunction

" Store the completion response that is shown as inline completion.
let s:current_completion_response = {}

function! s:HandleCompletionResponse(request, response)
  if s:current_completion_request != a:request
    return
  endif
  let s:ongoing_request_id = 0
  if (type(a:response) != v:t_dict) || !has_key(a:response, 'choices') ||
    \ (type(a:response.choices) != v:t_list)
    return
  endif
  call tabby#Dismiss()
  if (len(a:response.choices) == 0)
    return
  endif
  " Only support single choice completion for now
  let choice = a:response.choices[0]
  call tabby#virtual_text#Render(s:current_completion_request, choice)
  let s:current_completion_response = a:response
  
  call tabby#agent#PostEvent(#{
    \ type: "view",
    \ completion_id: a:response.id,
    \ choice_index: choice.index,
    \ })
endfunction

" Used as a buffer to store the text that should be inserted when user accepts
" the completion.
let s:text_to_insert = ''

function! tabby#ConsumeInsertion()
  let text = s:text_to_insert
  let s:text_to_insert = ''
  return text
endfunction

function! tabby#Accept(...)
  if s:current_completion_response == {}
    " keybindings fallback
    if a:0 < 1
      return "\<Ignore>"
    elseif type(a:1) == v:t_string
      return a:1
    elseif type(a:1) == v:t_func
      return call(a:1, [])
    endif
  endif

  let response = s:current_completion_response
  let choice = response.choices[0]
  if (type(choice.text) != v:t_string) || (len(choice.text) == 0)
    return
  endif
  let prefix_replace_chars = s:current_completion_request.position - choice.replaceRange.start 
  let suffix_replace_chars = choice.replaceRange.end - s:current_completion_request.position
  let s:text_to_insert = strcharpart(choice.text, prefix_replace_chars)
  let insertion = repeat("\<Del>", suffix_replace_chars) . "\<C-R>\<C-O>=tabby#ConsumeInsertion()\<CR>"
  
  if s:text_to_insert[-1:] == "\n"
    " Add a char and remove, workaround for insertion bug if ends with newline
    let s:text_to_insert .= "_"
    let insertion .= "\<BS>"
  endif
  
  call tabby#Dismiss()
  
  call tabby#agent#PostEvent(#{
    \ type: "select",
    \ completion_id: response.id,
    \ choice_index: choice.index,
    \ })

  return insertion
endfunction

function! tabby#Dismiss()
  let s:current_completion_response = {}
  call tabby#virtual_text#Clear()
endfunction

function! s:GetCompletionContext(is_manual)
  return #{
    \ filepath: expand('%:p'),
    \ language: s:GetLanguage(),
    \ text: join(getbufline('%', 1, '$'), "\n"),
    \ position: s:GetCursorPosition(),
    \ manually: a:is_manual,
    \ }
endfunction

" Count the number of characters from the beginning of the buffer to the cursor.
function! s:GetCursorPosition()
  let lines = getline(1, line('.') - 1)
  if col('.') > 1
    let lines += [strpart(getline(line('.')), 0, col('.') - 1)]
  else
    let lines += ['']
  endif
  return strchars(join(lines, "\n"))
endfunction

function! s:GetLanguage()
  let filetype = getbufvar('%', '&filetype')
  if has_key(g:tabby_filetype_dict, filetype)
    return g:tabby_filetype_dict[filetype]
  else
    return filetype
  endif
endfunction
