package com.tabbyml.intellijtabby.actions

import com.intellij.icons.AllIcons
import com.intellij.openapi.actionSystem.ActionUpdateThread
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.actionSystem.CommonDataKeys
import com.intellij.openapi.components.serviceOrNull
import com.tabbyml.intellijtabby.events.CombinedState
import com.tabbyml.intellijtabby.lsp.ConnectionService
import com.tabbyml.intellijtabby.lsp.protocol.StatusInfo
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.eclipse.lsp4j.ExecuteCommandParams

class CheckIssueDetail : AnAction() {
  private val scope = CoroutineScope(Dispatchers.IO)

  override fun actionPerformed(e: AnActionEvent) {
    val project = e.getRequiredData(CommonDataKeys.PROJECT)
    val combinedState = project.serviceOrNull<CombinedState>() ?: return
    val connectionService = project.serviceOrNull<ConnectionService>() ?: return

    scope.launch {
      val server = connectionService.getServerAsync() ?: return@launch
      val command = combinedState.state.agentStatus?.command ?: return@launch

      server.workspaceFeature.executeCommand(
        ExecuteCommandParams(
          command.command,
          command.arguments,
        )
      )
    }
  }

  override fun update(e: AnActionEvent) {
    e.presentation.isEnabled = e.project != null && e.getData(CommonDataKeys.PROJECT) != null
    val project = e.getData(CommonDataKeys.PROJECT) ?: return
    val combinedState = project.serviceOrNull<CombinedState>() ?: return

    e.presentation.isVisible = combinedState.state.agentStatus?.command != null
    e.presentation.icon = if (combinedState.state.agentStatus?.status == StatusInfo.Status.DISCONNECTED) {
      AllIcons.General.Error
    } else {
      AllIcons.General.Warning
    }
  }

  override fun getActionUpdateThread(): ActionUpdateThread {
    return ActionUpdateThread.BGT
  }
}