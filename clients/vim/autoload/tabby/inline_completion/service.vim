if exists('g:autoloaded_tabby_inline_completion_service')
  finish
endif
let g:autoloaded_tabby_inline_completion_service = 1

let g:tabby_inline_completion_trigger = get(g:, 'tabby_inline_completion_trigger', 'auto')
let g:tabby_inline_completion_insertion_leading_key = get(g:, 'tabby_inline_completion_insertion_leading_key', "\<C-R>\<C-O>=")

let s:current_request_context = {}
let s:current_request_id = 0

function! tabby#inline_completion#service#ClearCurrentIfNotMatch()
  let context = s:CreateInlineCompletionContext(v:false)
  if (has_key(s:current_request_context, 'buf') && s:current_request_context.buf == context.buf &&
    \ has_key(s:current_request_context, 'offset') && s:current_request_context.offset == context.offset &&
    \ has_key(s:current_request_context, 'modification') && s:current_request_context.modification == context.modification)
    return
  endif
  call tabby#inline_completion#service#Clear()
endfunction

function! tabby#inline_completion#service#TriggerOrDismiss()
  if s:current_completion_list != {}
    call tabby#inline_completion#service#Dismiss()
  else
    call tabby#inline_completion#service#Trigger(v:true)
  endif
  return ''
endfunction

function! tabby#inline_completion#service#Trigger(is_manually)
  if s:current_request_id != 0
    call g:tabby_inline_completion_source.CancelRequest(s:current_request_id)
  endif
  if g:tabby_inline_completion_trigger != 'auto' && !a:is_manually
    return
  endif
  let params = s:CreateInlineCompletionContext(a:is_manually)
  let s:current_request_context = params
  let OnResponse = { result -> s:HandleCompletionResponse(params, result) }
  let s:current_request_id = g:tabby_inline_completion_source.RequestInlineCompletion(params, OnResponse)
endfunction

function! s:CreateInlineCompletionContext(is_manually)
  return #{
    \ buf: bufnr(),
    \ offset: tabby#inline_completion#utils#GetCurrentOffset(),
    \ modification: getbufvar('%', '&modified'),
    \ trigger_kind: a:is_manually ? 1 : 2,
    \ }
endfunction

" Store the current completion list.
let s:current_completion_list = {}
let s:current_completion_item_index = 0
let s:current_completion_item_display_at = 0
let s:current_completion_item_display_eventid = ''

function! s:HandleCompletionResponse(params, result)
  if s:current_request_context != a:params
    return
  endif
  let s:ongoing_request_id = 0
  if (type(a:result) != v:t_dict) || !has_key(a:result, 'items') ||
    \ (type(a:result.items) != v:t_list)
    return
  endif
  call tabby#inline_completion#service#Dismiss()
  if (len(a:result.items) == 0)
    return
  endif
  let list = a:result
  let s:current_completion_list = list
  " FIXME(@icycodes): Only support single choice completion for now
  let s:current_completion_item_index = 0
  let item = list.items[s:current_completion_item_index]
  call tabby#inline_completion#virtual_text#Render(item)
  let s:current_completion_item_display_at = tabby#inline_completion#utils#GetTimestamp()

  if (has_key(item, 'data') && has_key(item.data, 'eventId') && has_key(item.data.eventId, 'completionId'))
    let cmpl_id = item.data.eventId.completionId
    let choice_index = item.data.eventId.choiceIndex
    let raw_cmpl_id = substitute(cmpl_id, 'cmpl-', '', '')
    let s:current_completion_item_display_eventid = 'view-' . raw_cmpl_id . '-at-' . string(s:current_completion_item_display_at)

    call g:tabby_inline_completion_source.NotifyEvent(#{
      \ type: "view",
      \ eventId: #{
        \ completionId: cmpl_id,
        \ choiceIndex: choice_index,
      \ },
      \ viewId: s:current_completion_item_display_eventid,
      \ })
  else
    let s:current_completion_item_display_eventid = ''
  endif
endfunction

" Used as a buffer to store the text that should be inserted when user accepts
" the completion.
let s:text_to_insert = ''

function! tabby#inline_completion#service#ConsumeInsertion()
  let text = s:text_to_insert
  let s:text_to_insert = ''
  return text
endfunction

function! tabby#inline_completion#service#Accept(...)
  if s:current_completion_list == {}
    " keybindings fallback
    if a:0 < 1
      return "\<Ignore>"
    elseif type(a:1) == v:t_string
      return a:1
    elseif type(a:1) == v:t_func
      return call(a:1, [])
    endif
  endif

  let accept_at = tabby#inline_completion#utils#GetTimestamp()
  let list = s:current_completion_list
  let item = list.items[s:current_completion_item_index]
  if (type(item.insertText) != v:t_string) || (len(item.insertText) == 0)
    return
  endif
  let char_count_col = tabby#inline_completion#utils#GetCharCountFromCol()
  let prefix_replace_chars = char_count_col - item.range.start.character
  let suffix_replace_chars = item.range.end.character - char_count_col
  let s:text_to_insert = strcharpart(item.insertText, prefix_replace_chars)
  let insertion = repeat("\<Del>", suffix_replace_chars) . g:tabby_inline_completion_insertion_leading_key . "tabby#inline_completion#service#ConsumeInsertion()\<CR>"
  
  if s:text_to_insert[-1:] == "\n"
    " Add a char and remove, workaround for insertion bug if ends with newline
    let s:text_to_insert .= "_"
    let insertion .= "\<BS>"
  endif

  if (has_key(item, 'data') && has_key(item.data, 'eventId') && has_key(item.data.eventId, 'completionId'))
    let cmpl_id = item.data.eventId.completionId
    let choice_index = item.data.eventId.choiceIndex

    call g:tabby_inline_completion_source.NotifyEvent(#{
      \ type: "select",
      \ eventId: #{
        \ completionId: cmpl_id,
        \ choiceIndex: choice_index,
      \ },
      \ viewId: s:current_completion_item_display_eventid,
      \ elapsed: accept_at - s:current_completion_item_display_at,
      \ })
  endif

  call tabby#inline_completion#service#Clear()
  return insertion
endfunction

function! tabby#inline_completion#service#Dismiss()
  if s:current_completion_list == {}
    return
  endif

  let dismiss_at = tabby#inline_completion#utils#GetTimestamp()
  let list = s:current_completion_list
  let item = list.items[s:current_completion_item_index]

  if (has_key(item, 'data') && has_key(item.data, 'eventId') && has_key(item.data.eventId, 'completionId'))
    let cmpl_id = item.data.eventId.completionId
    let choice_index = item.data.eventId.choiceIndex

    call g:tabby_inline_completion_source.NotifyEvent(#{
      \ type: "dismiss",
      \ eventId: #{
        \ completionId: cmpl_id,
        \ choiceIndex: choice_index,
      \ },
      \ viewId: s:current_completion_item_display_eventid,
      \ elapsed: dismiss_at - s:current_completion_item_display_at,
      \ })
  endif

  call tabby#inline_completion#service#Clear()
endfunction

function! tabby#inline_completion#service#Clear()
  let s:current_request_context = {}
  if s:current_request_id != 0
    call g:tabby_inline_completion_source.CancelRequest(s:current_request_id)
  endif
  let s:current_request_id = 0

  let s:current_completion_list = {}
  let s:current_completion_item_index = 0
  let s:current_completion_item_display_at = 0
  let s:current_completion_item_display_eventid = ''
  call tabby#inline_completion#virtual_text#Clear()
endfunction
