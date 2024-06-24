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
      val offset = caret?.offset ?: return
      inlineCompletionHandler.doExecute(editor, offset, inlineCompletionService)
    }

    override fun isEnabledForCaret(editor: Editor, caret: Caret, dataContext: DataContext?): Boolean {
      val inlineCompletionService = editor.project?.service<InlineCompletionService>() ?: return false
      return inlineCompletionService.isInlineCompletionVisibleAt(
        editor,
        caret.offset
      ) && inlineCompletionHandler.isEnabled(editor, caret.offset, inlineCompletionService)
    }
  }), HasPriority {
  override val priority: Int = 1
}