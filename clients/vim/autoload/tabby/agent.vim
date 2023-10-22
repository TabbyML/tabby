" Implementation of agent interface

if exists('g:autoloaded_tabby_agent')
  finish
endif
let g:autoloaded_tabby_agent = 1

"   Stores the job of the current tabby agent node process
let s:tabby = 0

"   Stores the status of the tabby agent
let s:tabby_status = 'notInitialized'

"   Stores the name of issues if any
let s:tabby_issues = []

function! tabby#agent#Status()
  return s:tabby_status
endfunction

function! tabby#agent#Issues()
  return s:tabby_issues
endfunction

function! tabby#agent#Open(command)
  if type(s:tabby) != v:t_number || s:tabby != 0
    return
  endif

  let s:tabby = tabby#job#Start(a:command, #{
    \ out_cb: { _, data -> s:OnNotification(data) },
    \ err_cb: { _, data -> s:OnError(data) },
    \ exit_cb: { _ -> s:OnExit() },
    \ })

  call tabby#agent#Initialize()
endfunction

function! s:OnNotification(data)
  if (type(a:data) == v:t_dict) && has_key(a:data, 'event')
    if  a:data.event == 'statusChanged'
      let s:tabby_status = a:data.status
    elseif a:data.event == 'issuesUpdated'
      let s:tabby_issue = a:data.issues
    endif
  endif
endfunction

function! s:OnError(data)
  " For Debug
  " echoerr "OnError: " . string(a:data)
endfunction

function! s:OnExit()
  let s:tabby = {}
  let s:tabby_status = 'exited'
endfunction

function! tabby#agent#Close()
  if type(s:tabby) == v:t_number && s:tabby == 0
    return
  endif
  call tabby#job#Stop(s:tabby)
  let s:tabby = {}
  let s:tabby_status = 'exited'
endfunction

function! tabby#agent#Initialize()
  if type(s:tabby) == v:t_number && s:tabby == 0
    return
  endif
  call tabby#job#Send(s:tabby, #{
    \ func: 'initialize',
    \ args: [#{
      \ clientProperties: s:GetClientProperties(),
      \ }],
    \ })
endfunction

function! tabby#agent#RequestAuthUrl(OnResponse)
  if type(s:tabby) == v:t_number && s:tabby == 0
    return
  endif
  call tabby#job#Send(s:tabby, #{
    \ func: 'requestAuthUrl',
    \ args: [],
    \ }, #{
    \ callback: { _, data -> a:OnResponse(data) },
    \ })
endfunction

function! tabby#agent#WaitForAuthToken(code)
  if type(s:tabby) == v:t_number && s:tabby == 0
    return
  endif
  call tabby#job#Send(s:tabby, #{
    \ func: 'waitForAuthToken',
    \ args: [a:code],
    \ })
endfunction

function! tabby#agent#ProvideCompletions(request, OnResponse)
  if type(s:tabby) == v:t_number && s:tabby == 0
    return
  endif
  let requestId = tabby#job#Send(s:tabby, #{
    \ func: 'provideCompletions',
    \ args: [a:request, { "signal": v:true }],
    \ }, #{
    \ callback: { _, data -> a:OnResponse(data) },
    \ })
  return requestId
endfunction

function! tabby#agent#CancelRequest(requestId)
  if type(s:tabby) == v:t_number && s:tabby == 0
    return
  endif
  call tabby#job#Send(s:tabby, #{
    \ func: 'cancelRequest',
    \ args: [a:requestId],
    \ })
endfunction

function! tabby#agent#PostEvent(event)
  if type(s:tabby) == v:t_number && s:tabby == 0
    return
  endif
  call tabby#job#Send(s:tabby, #{
    \ func: 'postEvent',
    \ args: [a:event],
    \ })
endfunction

function! s:GetClientProperties()
  let version_output = execute('version')
  let client = split(version_output, "\n")[0]
  let name = split(client, ' ')[0]
  return #{
    \ user: #{
      \ vim: #{
        \ triggerMode: g:tabby_trigger_mode
      \ }
    \ },
    \ session: #{
      \ client: client,
      \ ide: #{
        \ name: name,
        \ version: client,
      \ },
      \ tabby_plugin: #{
        \ name: 'TabbyML/vim-tabby',
        \ version: g:tabby_version,
      \ },
    \ }
  \ }
endfunction
