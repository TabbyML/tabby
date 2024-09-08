if exists('g:autoloaded_tabby_lsp_nvim_lsp')
  finish
endif
let g:autoloaded_tabby_lsp_nvim_lsp = 1

" Setup

function! tabby#lsp#nvim_lsp#Setup()
  if !has('nvim')
    return v:false
  endif
  return v:lua.require'tabby'.lsp.nvim_lsp.setup()
endfunction

let s:client = {}

function! tabby#lsp#nvim_lsp#GetClient()
  return s:client
endfunction

" CancelReqeust

function! s:client.CancelReqeust(id)
  return v:lua.require'tabby'.lsp.nvim_lsp.cancel_request(a:id)
endfunction

" RequestInlineCompletion

let s:request_inline_completion_callbacks = {}

function! s:client.RequestInlineCompletion(params, callback)
  let request_id = v:lua.require'tabby'.lsp.nvim_lsp.request_inline_completion(a:params)
  let s:request_inline_completion_callbacks[request_id] = a:callback
endfunction

function! tabby#lsp#nvim_lsp#CallInlineCompletionCallback(request_id, result)
  if has_key(s:request_inline_completion_callbacks, a:request_id)
    call s:request_inline_completion_callbacks[a:request_id](a:result)
    unlet s:request_inline_completion_callbacks[a:request_id]
  endif
endfunction

" NotifyEvent

function! s:client.NotifyEvent(params)
  return v:lua.require'tabby'.lsp.nvim_lsp.notify_event(a:params)
endfunction

