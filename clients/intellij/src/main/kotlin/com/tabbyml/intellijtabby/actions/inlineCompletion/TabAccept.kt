package com.tabbyml.intellijtabby.actions.inlineCompletion

import com.intellij.openapi.editor.Caret
import com.intellij.openapi.editor.Editor
import com.tabbyml.intellijtabby.completion.InlineCompletionService

class TabAccept : InlineCompletionAction(object : InlineCompletionActionHandler {
  override fun doExecute(editor: Editor, caret: Caret?, inlineCompletionService: InlineCompletionService) {
    inlineCompletionService.accept(editor, caret?.offset, InlineCompletionService.AcceptType.FULL_COMPLETION)
  }

  override fun isEnabledForCaret(
    editor: Editor,
    caret: Caret,
    inlineCompletionService: InlineCompletionService
  ): Boolean {
    return !inlineCompletionService.isInlineCompletionStartWithIndentation()
  }
})
