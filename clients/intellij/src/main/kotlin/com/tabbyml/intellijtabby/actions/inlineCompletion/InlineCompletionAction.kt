package com.tabbyml.intellijtabby.actions.inlineCompletion

import com.intellij.openapi.actionSystem.DataContext
import com.intellij.openapi.components.service
import com.intellij.openapi.editor.Caret
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.actionSystem.EditorAction
import com.intellij.openapi.editor.actionSystem.EditorActionHandler
import com.tabbyml.intellijtabby.actionPromoter.HasPriority
import com.tabbyml.intellijtabby.completion.InlineCompletionService

abstract class InlineCompletionAction(private val inlineCompletionHandler: InlineCompletionActionHandler) :
  EditorAction(object : EditorActionHandler() {
    override fun doExecute(editor: Editor, caret: Caret?, dataContext: DataContext?) {
      val inlineCompletionService = editor.project?.service<InlineCompletionService>() ?: return
      inlineCompletionHandler.doExecute(editor, caret, inlineCompletionService)
    }

    override fun isEnabledForCaret(editor: Editor, caret: Caret, dataContext: DataContext?): Boolean {
      val inlineCompletionService = editor.project?.service<InlineCompletionService>() ?: return false
      return inlineCompletionService.isInlineCompletionVisibleAt(
        editor,
        caret.offset
      ) && inlineCompletionHandler.isEnabledForCaret(editor, caret, inlineCompletionService)
    }
  }), HasPriority {
  override val priority: Int = 1
}