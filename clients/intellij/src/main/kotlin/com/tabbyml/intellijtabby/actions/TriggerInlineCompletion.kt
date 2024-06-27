package com.tabbyml.intellijtabby.actions

import com.intellij.openapi.actionSystem.ActionUpdateThread
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.actionSystem.CommonDataKeys
import com.intellij.openapi.components.service
import com.tabbyml.intellijtabby.actionPromoter.HasPriority
import com.tabbyml.intellijtabby.completion.InlineCompletionService


class TriggerInlineCompletion : AnAction(), HasPriority {
  override fun actionPerformed(e: AnActionEvent) {
    val inlineCompletionService = e.getRequiredData(CommonDataKeys.PROJECT).service<InlineCompletionService>()
    val editor = e.getRequiredData(CommonDataKeys.EDITOR)
    val offset = editor.caretModel.primaryCaret.offset
    inlineCompletionService.provideInlineCompletion(editor, offset, manually = true)
  }

  override fun update(e: AnActionEvent) {
    e.presentation.isEnabled =
      e.project != null && e.getData(CommonDataKeys.PROJECT) != null && e.getData(CommonDataKeys.EDITOR) != null
  }

  override fun getActionUpdateThread(): ActionUpdateThread {
    return ActionUpdateThread.BGT
  }

  override val priority: Int = 1
}
