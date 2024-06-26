package com.tabbyml.intellijtabby.actions.inlineCompletion

import com.intellij.openapi.editor.Caret
import com.intellij.openapi.editor.Editor
import com.tabbyml.intellijtabby.completion.InlineCompletionService

class CyclePrevious : InlineCompletionAction(object : InlineCompletionActionHandler {
  override fun doExecute(editor: Editor, caret: Caret?, inlineCompletionService: InlineCompletionService) {
    inlineCompletionService.cycle(editor, caret?.offset, InlineCompletionService.CycleDirection.PREVIOUS)
  }
})
