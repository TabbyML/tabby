package com.tabbyml.intellijtabby.completion

import com.tabbyml.intellijtabby.lsp.protocol.CompletionEventId

data class InlineCompletionList(
  val isIncomplete: Boolean,
  val items: List<InlineCompletionItem>,
)

data class InlineCompletionItem(
  val insertText: String,
  val replaceRange: Range,
  val data: Data? = null,
) {
  data class Range(
    val start: Int,
    val end: Int,
  )

  data class Data(
    val eventId: CompletionEventId? = null
  )
}