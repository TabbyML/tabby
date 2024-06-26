package com.tabbyml.intellijtabby.actions.inlineCompletion

import com.intellij.openapi.editor.Caret
import com.intellij.openapi.editor.Editor
import com.tabbyml.intellijtabby.completion.InlineCompletionService

class Dismiss : InlineCompletionAction(object : InlineCompletionActionHandler {
  override fun doExecute(editor: Editor, caret: Caret?, inlineCompletionService: InlineCompletionService) {
    inlineCompletionService.dismiss()
  }
}) {
  override val priority = -1
}

