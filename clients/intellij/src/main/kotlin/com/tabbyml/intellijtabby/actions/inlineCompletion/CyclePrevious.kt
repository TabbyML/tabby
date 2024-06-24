package com.tabbyml.intellijtabby.actions.inlineCompletion

import com.intellij.openapi.editor.Editor
import com.tabbyml.intellijtabby.completion.InlineCompletionService

class CyclePrevious : InlineCompletionAction(object : InlineCompletionActionHandler {
  override fun doExecute(editor: Editor, offset: Int, inlineCompletionService: InlineCompletionService) {
    inlineCompletionService.cycle(editor, offset, InlineCompletionService.CycleDirection.PREVIOUS)
  }
})
