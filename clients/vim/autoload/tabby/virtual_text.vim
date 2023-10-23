" Handles virtual text (aka ghost text) for rendering inline completion

if exists('g:autoloaded_tabby_virtual_text')
  finish
endif
let g:autoloaded_tabby_virtual_text = 1

let s:vim = exists('*prop_type_add')
let s:nvim = !s:vim && has('nvim') && exists('*nvim_buf_set_extmark')

function! tabby#virtual_text#Check()
  return #{
    \ ok: s:vim || s:nvim,
    \ message: 'Tabby requires Vim 9.0.0534+ with +textprop feature support, or NeoVim 0.6.0+.',
    \ }
endfunction

function! tabby#virtual_text#Init()
  hi def TabbyCompletion guifg=#808080 ctermfg=245
  hi def TabbyCompletionReplaceRange guifg=#303030 ctermfg=236 guibg=#808080 ctermbg=245
  if s:vim
    let s:prop_type_completion = 'TabbyCompletion'
    if prop_type_get(s:prop_type_completion) != {}
      call prop_type_delete(s:prop_type_completion)
    endif
    call prop_type_add(s:prop_type_completion, #{
      \ highlight: 'TabbyCompletion',
      \ priority: 99,
      \ combine: 0,
      \ override: 1,
      \ })

    let s:prop_type_replace = 'TabbyCompletionReplaceRange'
    if prop_type_get(s:prop_type_replace) != {}
      call prop_type_delete(s:prop_type_replace)
    endif
    call prop_type_add(s:prop_type_replace, #{
      \ highlight: 'TabbyCompletionReplaceRange',
      \ priority: 99,
      \ combine: 0,
      \ override: 1,
      \ })
  endif
  if s:nvim
    let s:nvim_namespace = nvim_create_namespace('TabbyCompletion')
    let s:nvim_highlight_completion = 'TabbyCompletion'
    let s:nvim_highlight_replace = 'TabbyCompletionReplaceRange'
  endif
endfunction

function! tabby#virtual_text#Render(request, choice)
  if (type(a:choice.text) != v:t_string) || (len(a:choice.text) == 0)
    return
  endif
  let prefix_replace_chars = a:request.position - a:choice.replaceRange.start 
  let suffix_replace_chars = a:choice.replaceRange.end - a:request.position
  let text = strcharpart(a:choice.text, prefix_replace_chars)
  if len(text) == 0
    return
  endif
  let current_line_suffix = strpart(getline('.'), col('.') - 1)
  if strchars(current_line_suffix) < suffix_replace_chars
    return
  endif
  let text_lines = split(text, "\n")
  " split will not give an empty line if text starts or ends with "\n"
  if text[:0] == "\n"
    call insert(text_lines, '')
  endif
  if text[-1:] == "\n"
    call add(text_lines, '')
  endif
  " FIXME: no replace range processing for nvim for now, we need 
  "  feat `virt_text_pos: "inline"` after nvim 0.10.0
  if s:nvim
    let text_lines[-1] .= strcharpart(current_line_suffix, suffix_replace_chars)
    call s:AddInlay(text_lines[0], col('.'))
    if len(text_lines) > 1
      call s:AddLinesBelow(text_lines[1:])
    endif
    return
  endif
  " Replace range processing for vim
  if suffix_replace_chars == 0
    call s:AddInlay(text_lines[0], col('.'))
    if len(text_lines) > 1
      if strchars(current_line_suffix) > 0
        call s:MarkReplaceRange(range(col('.'), col('.') + len(current_line_suffix)))
        let text_lines[-1] .= current_line_suffix
      endif
      call s:AddLinesBelow(text_lines[1:])
    endif
  elseif suffix_replace_chars == 1
    let replace_char = strcharpart(current_line_suffix, 0, 1)
    let inlay = ''
    if strchars(text_lines[0]) > 0 && stridx(text_lines[0], replace_char) != 0
      let inlay = split(text_lines[0], replace_char)[0]
    endif
    call s:AddInlay(inlay, col('.'))
    if inlay != text_lines[0]
      let inlay_suffix = strpart(text_lines[0], len(inlay) + len(replace_char))
      call s:AddInlay(inlay_suffix, col('.') + len(replace_char))
    endif
    if len(text_lines) > 1
      if strchars(current_line_suffix) > 0
        let range_start = col('.')
        if inlay != text_lines[0]
          let range_start += len(replace_char)
        endif
        call s:MarkReplaceRange(range(range_start, col('.') + len(current_line_suffix)))
        let text_lines[-1] .= strcharpart(current_line_suffix, 1)
      endif
      call s:AddLinesBelow(text_lines[1:])
    endif
  else
    let replace_char = strcharpart(current_line_suffix, 0, suffix_replace_chars)
    call s:AddInlay(text_lines[0], col('.'))
    call s:MarkReplaceRange(range(col('.'), col('.') + len(replace_char)))
    if len(text_lines) > 1
      if strchars(current_line_suffix) > suffix_replace_chars
        call s:MarkReplaceRange(range(col('.') + len(replace_char), col('.') + len(current_line_suffix)))
        let text_lines[-1] .= strcharpart(current_line_suffix, suffix_replace_chars)
      endif
      call s:AddLinesBelow(text_lines[1:])
    endif
  endif
endfunction

function! s:AddInlay(inlay, column)
  if s:vim
    if len(a:inlay) > 0
      call prop_add(line('.'), a:column, #{
        \ type: s:prop_type_completion,
        \ text: a:inlay,
        \ })
    endif
  endif
  if s:nvim
    if len(a:inlay) > 0
      " FIXME: using virt_text_pos: "inline" after nvim 0.10.0
      call nvim_buf_set_extmark(0, s:nvim_namespace, line('.') - 1, col('.') - 1, #{
        \ virt_text_win_col: virtcol('.') - 1,
        \ virt_text: [[a:inlay, s:nvim_highlight_completion]],
        \ })
    endif
  endif
endfunction

function! s:AddLinesBelow(lines_below)
  if s:vim
    for line_blow in a:lines_below
      let text = line_blow
      if len(text) == 0
        let text = ' '
      endif
      call prop_add(line('.'), 0, #{
        \ type: s:prop_type_completion,
        \ text: text,
        \ text_align: 'below',
        \ })
    endfor
  endif
  if s:nvim
    call nvim_buf_set_extmark(0, s:nvim_namespace, line('.') - 1, col('.') - 1, #{
      \ virt_lines: map(a:lines_below, { i, l -> [[l, s:nvim_highlight_completion]] })
      \ })
  endif
endfunction

function! s:MarkReplaceRange(replace_range)
  if s:vim
    call prop_add(line('.'), a:replace_range[0], #{
      \ type: s:prop_type_replace,
      \ length: len(a:replace_range),
      \ })
  endif
  if s:nvim
    call nvim_buf_add_highlight(0, s:nvim_namespace, s:nvim_highlight_replace, line('.') - 1, 
      \ a:replace_range[0] - 1, a:replace_range[-1])
  endif
endfunction

function! tabby#virtual_text#Clear()
  if s:vim
    call prop_remove(#{
      \ type: s:prop_type_completion,
      \ all: 1,
      \ })
    call prop_remove(#{
      \ type: s:prop_type_replace,
      \ all: 1,
      \ })
  endif
  if s:nvim
    call nvim_buf_clear_namespace(0, s:nvim_namespace, 0, -1)
  endif
endfunction
