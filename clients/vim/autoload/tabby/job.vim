" Handles IO with child processes jobs

if exists('g:autoloaded_tabby_job')
  finish
endif
let g:autoloaded_tabby_job = 1

let s:vim = exists('*job_start')
let s:nvim = !s:vim && has('nvim') && exists('*jobstart')

function! tabby#job#Check()
  return #{
    \ ok: s:vim || s:nvim,
    \ message: 'Tabby requires Vim 9.0+ with +job feature support, or NeoVim 0.6.0+.',
    \ }
endfunction

let s:nvim_job_map = {}

" Assume Json IO
" Options 'out_cb', 'err_cb', 'exit_cb' supported
" Return job id
function! tabby#job#Start(command, ...)
  let options = get(a:, 1, {})
  if s:vim
    let opt = #{
      \ in_mode: 'json',
      \ out_mode: 'json',
      \ }
    if has_key(options, 'out_cb')
      let opt.out_cb = options.out_cb
    endif
    if has_key(options, 'err_cb')
      let opt.err_cb = options.err_cb
    endif
    if has_key(options, 'exit_cb')
      let opt.exit_cb = options.exit_cb
    endif
    return job_start(a:command, opt)
  endif
  if s:nvim
    let id = jobstart(a:command, #{
      \ on_stdout: function('s:NvimHandleStdout'),
      \ on_stderr: function('s:NvimHandleStderr'),
      \ on_exit: function('s:NvimHandleExit'),
      \ })
    let s:nvim_job_map[id] = #{
      \ out_buffer: '',
      \ requests: {},
      \ }
    if has_key(options, 'out_cb')
      let s:nvim_job_map[id].out_cb = options.out_cb
    endif
    if has_key(options, 'err_cb')
      let s:nvim_job_map[id].err_cb = options.err_cb
    endif
    if has_key(options, 'exit_cb')
      let s:nvim_job_map[id].exit_cb = options.exit_cb
    endif
    return id
  endif
endfunction

function! tabby#job#Stop(job)
  if s:vim
    return job_stop(a:job)
  endif
  if s:nvim
    let ret = jobstop(a:job)
    if has_key(s:nvim_job_map, a:job)
      unlet s:nvim_job_map[a:job]
    endif
    return ret
  endif
endfunction

" Align to Vim's ch_sendexpr
" Options 'callback' supported
function! tabby#job#Send(job, data, ...)
  let options = get(a:, 1, {})
  let id = s:NextRequestId()
  if s:vim
    call ch_sendexpr(a:job, a:data, options)
  endif
  if s:nvim
    let request = [id, a:data]
    let s:nvim_job_map[a:job].requests[id] = {}
    if has_key(options, 'callback')
      let s:nvim_job_map[a:job].requests[id].callback = options.callback
    endif
    call chansend(a:job, json_encode(request) . "\n")
  endif
  return id
endfunction

let s:request_id = 0
function! s:NextRequestId()
  let s:request_id += 1
  return s:request_id
endfunction

function! s:NvimHandleStdout(job, data, event)
  if !has_key(s:nvim_job_map, a:job)
    return
  endif
  let buf = s:nvim_job_map[a:job].out_buffer
  for data_line in a:data
    let buf .= data_line
    try
      let decoded = json_decode(buf)
      let buf = ''
    catch
      continue
    endtry
    call s:NvimHandleOutDecoded(a:job, decoded)
  endfor
  let s:nvim_job_map[a:job].out_buffer = buf
endfunction

function! s:NvimHandleOutDecoded(job, decoded)
  if !has_key(s:nvim_job_map, a:job)
    return
  endif
  if type(a:decoded) != v:t_list || len(a:decoded) < 1 || (type(a:decoded[0]) != v:t_number)
    return
  endif
  let id = a:decoded[0]
  if len(a:decoded) >= 2
    let data = a:decoded[1]
  else
    let data = {}
  endif
  if (id > 0) && has_key(s:nvim_job_map[a:job].requests, id)
    let request = s:nvim_job_map[a:job].requests[id]
    if has_key(request, 'callback')
      call request.callback(a:job, data)
    endif
    unlet s:nvim_job_map[a:job].requests[id]
  else
    if has_key(s:nvim_job_map[a:job], 'out_cb')
      call s:nvim_job_map[a:job].out_cb(a:job, data)
    endif
  endif
endfunction

function! s:NvimHandleStderr(job, data, event)
  if !has_key(s:nvim_job_map, a:job)
    return
  endif
  if has_key(s:nvim_job_map[a:job], 'err_cb')
    call s:nvim_job_map[a:job].err_cb(a:job, join(a:data, "\n"))
  endif
endfunction

function! s:NvimHandleExit(job, status, event)
  if !has_key(s:nvim_job_map, a:job)
    return
  endif
  if has_key(s:nvim_job_map[a:job], 'exit_cb')
    call s:nvim_job_map[a:job].exit_cb(a:job, a:status)
  endif
endfunction
