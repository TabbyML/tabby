if exists('g:autoloaded_tabby_lsp')
  finish
endif
let g:autoloaded_tabby_lsp = 1

let g:tabby_agent_start_command = get(g:, 'tabby_agent_start_command', ['npx', 'tabby-agent', '--stdio'])
let g:tabby_lsp_client = get(g:, 'tabby_lsp_client', {})

function! tabby#lsp#Setup()
  if g:tabby_lsp_client == {} && tabby#lsp#nvim_lsp#Setup()
    let g:tabby_lsp_client = tabby#lsp#nvim_lsp#GetClient()
  endif
endfunction

function! tabby#lsp#CancelReqeust(id)
  if (has_key(g:tabby_lsp_client, 'CancelReqeust'))
    return g:tabby_lsp_client.CancelReqeust(a:params)
  endif
endfunction

function! tabby#lsp#RequestInlineCompletion(params, callback)
  if (has_key(g:tabby_lsp_client, 'RequestInlineCompletion'))
    return g:tabby_lsp_client.RequestInlineCompletion(a:params, a:callback)
  else
    return 0
  endif
endfunction

function! tabby#lsp#NotifyEvent(params)
  if (has_key(g:tabby_lsp_client, 'NotifyEvent'))
    return g:tabby_lsp_client.NotifyEvent(a:params)
  endif
endfunction

function! tabby#lsp#RequestStatus(params, callback)
  if (has_key(g:tabby_lsp_client, 'RequestStatus'))
    return g:tabby_lsp_client.RequestStatus(a:params, a:callback)
  else
    return 0
  endif
endfunction
