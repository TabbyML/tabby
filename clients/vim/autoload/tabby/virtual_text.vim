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
    \}
endfunction

function! tabby#virtual_text#Init()
  hi def TabbyCompletion guifg=#808080 ctermfg=8
  if s:vim
    let s:prop_type = 'TabbyCompletion'
    if prop_type_get(s:prop_type) != v:null
      call prop_type_delete(s:prop_type)
    endif
    call prop_type_add(s:prop_type, {'highlight': 'TabbyCompletion'})
  endif
  if s:nvim
    let s:nvim_namespace = nvim_create_namespace('TabbyCompletion')
    let s:nvim_highlight = 'TabbyCompletion'
    let s:nvim_extmark_id = 1
  endif
endfunction

function! tabby#virtual_text#Show(lines)
  if len(a:lines) == 0
    return
  endif
  if s:vim
    " virtual text requires 9.0.0534
    " append line with a space to avoid empty line, which will result in unexpected behavior
    call prop_add(line('.'), col('.'), #{
      \ type: s:prop_type,
      \ text: a:lines[0] . ' ', 
      \ })
    for line in a:lines[1:]
      call prop_add(line('.'), 0, #{
        \ type: s:prop_type,
        \ text: line . ' ',
        \ text_align: 'below',
        \ })
    endfor
  endif
  if s:nvim
    let opt = #{
      \ id: s:nvim_extmark_id,
      \ virt_text_win_col: virtcol('.') - 1,
      \ virt_text: [[a:lines[0], s:nvim_highlight]],
      \}
    if len(a:lines) > 1
      let opt.virt_lines = map(a:lines[1:], { i, l -> [[l, s:nvim_highlight]] })
    endif
    call nvim_buf_set_extmark(0, s:nvim_namespace, line('.') - 1, col('.') - 1, opt)
  endif
endfunction

function! tabby#virtual_text#Clear()
  if s:vim
    call prop_remove(#{
      \ type: s:prop_type,
      \ all: v:true,
      \ })
  endif
  if s:nvim
    call nvim_buf_del_extmark(0, s:nvim_namespace, s:nvim_extmark_id)
  endif
endfunction
