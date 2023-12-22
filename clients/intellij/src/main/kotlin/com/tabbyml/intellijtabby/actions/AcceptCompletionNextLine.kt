package com.tabbyml.intellijtabby.actions

import com.intellij.openapi.actionSystem.DataContext
import com.intellij.openapi.components.service
import com.intellij.openapi.editor.Caret
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.actionSystem.EditorAction
import com.intellij.openapi.editor.actionSystem.EditorActionHandler
import com.tabbyml.intellijtabby.editor.InlineCompletionService

class AcceptCompletionNextLine : EditorAction(object : EditorActionHandler() {
  val inlineCompletionService = service<InlineCompletionService>()

  override fun doExecute(editor: Editor, caret: Caret?, dataContext: DataContext?) {
    inlineCompletionService.accept(InlineCompletionService.AcceptType.NEXT_LINE)
  }

  override fun isEnabledForCaret(editor: Editor, caret: Caret, dataContext: DataContext?): Boolean {
    return editor == inlineCompletionService.shownInlineCompletion?.editor
        && caret.offset == inlineCompletionService.shownInlineCompletion?.offset
  }
}), HasPriority {
  override val priority: Int = 1
}
