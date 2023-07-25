package com.tabbyml.intellijtabby.actions

import com.intellij.openapi.actionSystem.ActionUpdateThread
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.actionSystem.CommonDataKeys
import com.intellij.openapi.application.invokeLater
import com.intellij.openapi.components.service
import com.tabbyml.intellijtabby.agent.AgentService
import com.tabbyml.intellijtabby.editor.InlineCompletionService


class TriggerCompletion : AnAction() {
  override fun actionPerformed(e: AnActionEvent) {
    val agentService = service<AgentService>()
    val inlineCompletionService = service<InlineCompletionService>()
    val editor = e.getRequiredData(CommonDataKeys.EDITOR)
    val file = e.getRequiredData(CommonDataKeys.PSI_FILE)
    val offset = editor.caretModel.primaryCaret.offset

    inlineCompletionService.dismiss()
    agentService.getCompletion(editor, file, offset)?.thenAccept {
      invokeLater {
        inlineCompletionService.show(editor, offset, it)
      }
    }
  }

  override fun update(e: AnActionEvent) {
    val inlineCompletionService = service<InlineCompletionService>()
    e.presentation.isEnabled = e.project != null
        && e.getData(CommonDataKeys.EDITOR) != null
        && e.getData(CommonDataKeys.PSI_FILE) != null
        && inlineCompletionService.currentText == null
  }

  override fun getActionUpdateThread(): ActionUpdateThread {
    return ActionUpdateThread.BGT
  }
}
