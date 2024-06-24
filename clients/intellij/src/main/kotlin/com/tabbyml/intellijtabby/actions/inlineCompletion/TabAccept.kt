package com.tabbyml.intellijtabby.actions.inlineCompletion

import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.editor.Editor
import com.tabbyml.intellijtabby.completion.InlineCompletionService

class TabAccept : InlineCompletionAction(object : InlineCompletionActionHandler {
  override fun doExecute(editor: Editor, offset: Int, inlineCompletionService: InlineCompletionService) {
    inlineCompletionService.accept(editor, offset, InlineCompletionService.AcceptType.FULL_COMPLETION)
  }

  override fun isEnabled(editor: Editor, offset: Int, inlineCompletionService: InlineCompletionService): Boolean {
    return !inlineCompletionService.isInlineCompletionStartWithIndentation()
  }
}) {
  override fun update(e: AnActionEvent) {
    e.presentation.isVisible = false
  }
}
