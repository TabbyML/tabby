package com.tabbyml.intellijtabby.actions

import com.intellij.openapi.actionSystem.ActionUpdateThread
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.actionSystem.CommonDataKeys
import com.intellij.openapi.components.service
import com.tabbyml.intellijtabby.editor.InlineCompletionService


class AcceptCompletionNextLine : AnAction() {
  override fun actionPerformed(e: AnActionEvent) {
    val inlineCompletionService = service<InlineCompletionService>()
    inlineCompletionService.accept(InlineCompletionService.AcceptType.NEXT_LINE)
  }
  
  override fun update(e: AnActionEvent) {
    val inlineCompletionService = service<InlineCompletionService>()
    e.presentation.isEnabled = e.project != null
        && e.getData(CommonDataKeys.EDITOR) != null
        && inlineCompletionService.shownInlineCompletion != null
        && e.getData(CommonDataKeys.EDITOR) == inlineCompletionService.shownInlineCompletion?.editor
  }

  override fun getActionUpdateThread(): ActionUpdateThread {
    return ActionUpdateThread.EDT
  }
}
