package com.tabbyml.intellijtabby.actions.inlineCompletion

import com.intellij.openapi.editor.Caret
import com.intellij.openapi.editor.Editor
import com.tabbyml.intellijtabby.completion.InlineCompletionService

interface InlineCompletionActionHandler {
  fun doExecute(editor: Editor, caret: Caret?, inlineCompletionService: InlineCompletionService)
  fun isEnabledForCaret(editor: Editor, caret: Caret, inlineCompletionService: InlineCompletionService): Boolean {
    return true
  }
}