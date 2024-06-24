package com.tabbyml.intellijtabby.actions.inlineCompletion

import com.intellij.openapi.editor.Editor
import com.tabbyml.intellijtabby.completion.InlineCompletionService

interface InlineCompletionActionHandler {
  fun doExecute(editor: Editor, offset: Int, inlineCompletionService: InlineCompletionService)
  fun isEnabled(editor: Editor, offset: Int, inlineCompletionService: InlineCompletionService): Boolean {
    return true
  }
}