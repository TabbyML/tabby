package com.tabbyml.intellijtabby.actions

import com.intellij.openapi.actionSystem.ActionUpdateThread
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.actionSystem.CommonDataKeys
import com.intellij.openapi.components.service
import com.tabbyml.intellijtabby.agent.AgentService
import com.tabbyml.intellijtabby.editor.InlineCompletionService


class TriggerCompletion : AnAction() {
  override fun actionPerformed(e: AnActionEvent) {
    val agentService = service<AgentService>()
    val inlineCompletionService = service<InlineCompletionService>()
    val editor = e.getRequiredData(CommonDataKeys.EDITOR)
    val offset = editor.caretModel.primaryCaret.offset

    inlineCompletionService.dismiss()
    agentService.getCompletion(editor, offset)?.thenAccept {
      inlineCompletionService.show(editor, offset, it)
    }
  }

  override fun update(e: AnActionEvent) {
    e.presentation.isEnabled = e.project != null
        && e.getData(CommonDataKeys.EDITOR) != null
  }

  override fun getActionUpdateThread(): ActionUpdateThread {
    return ActionUpdateThread.BGT
  }
}
