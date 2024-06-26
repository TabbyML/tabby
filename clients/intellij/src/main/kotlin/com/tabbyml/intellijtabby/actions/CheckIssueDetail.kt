package com.tabbyml.intellijtabby.actions

import com.intellij.icons.AllIcons
import com.intellij.openapi.actionSystem.*
import com.intellij.openapi.application.invokeLater
import com.intellij.openapi.components.service
import com.intellij.openapi.ui.Messages
import com.intellij.openapi.ui.popup.JBPopupFactory
import com.tabbyml.intellijtabby.events.CombinedState
import com.tabbyml.intellijtabby.lsp.ConnectionService
import com.tabbyml.intellijtabby.lsp.protocol.IssueDetailParams
import com.tabbyml.intellijtabby.lsp.protocol.IssueName
import com.tabbyml.intellijtabby.settings.SettingsService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.future.await
import kotlinx.coroutines.launch

class CheckIssueDetail : AnAction() {
  override fun actionPerformed(e: AnActionEvent) {
    val project = e.getRequiredData(CommonDataKeys.PROJECT)
    val combinedState = project.service<CombinedState>()
    val issueName = combinedState.state.agentIssue ?: return

    val settings = service<SettingsService>()
    val connectionService = project.service<ConnectionService>()
    val scope = CoroutineScope(Dispatchers.IO)

    scope.launch {
      val server = connectionService.getServerAsync() ?: return@launch
      val detail = server.agentFeature.getIssueDetail(
        IssueDetailParams(
          name = issueName, helpMessageFormat = IssueDetailParams.HelpMessageFormat.HTML
        )
      ).await() ?: return@launch
      if (detail.name == IssueName.CONNECTION_FAILED) {
        invokeLater {
          val selected = Messages.showDialog(
            "<html>${detail.helpMessage}</html>",
            "Cannot Connect to Tabby Server",
            arrayOf("OK", "Online Help"),
            0,
            Messages.getErrorIcon(),
          )
          when (selected) {
            0 -> {
              // OK
            }

            1 -> {
              // Online Help
              showOnlineHelp(e)
            }
          }
        }
      } else {
        val title = when (detail.name) {
          IssueName.SLOW_COMPLETION_RESPONSE_TIME -> "Completion Requests Appear to Take Too Much Time"
          IssueName.HIGH_COMPLETION_TIMEOUT_RATE -> "Most Completion Requests Timed Out"
          else -> return@launch
        }
        invokeLater {
          val selected = Messages.showDialog(
            "<html>${detail.helpMessage}</html>",
            title,
            arrayOf("OK", "Online Help", "Don't Show Again"),
            0,
            Messages.getWarningIcon(),
          )
          when (selected) {
            0 -> {
              // OK
            }

            1 -> {
              // Online Help
              showOnlineHelp(e)
            }

            2 -> {
              // Don't Show Again
              settings.notificationsMuted += listOf("completionResponseTimeIssues")
              settings.notifyChanges(project)
            }
          }
        }
      }
    }
  }

  private fun showOnlineHelp(e: AnActionEvent) {
    e.project?.let {
      invokeLater {
        val actionManager = ActionManager.getInstance()
        val actionGroup = actionManager.getAction("Tabby.OpenOnlineHelp") as ActionGroup
        val popup = JBPopupFactory.getInstance().createActionGroupPopup(
          "Online Help",
          actionGroup,
          e.dataContext,
          false,
          null,
          10,
        )
        popup.showCenteredInCurrentWindow(it)
      }
    }
  }

  override fun update(e: AnActionEvent) {
    e.presentation.isEnabled = e.project != null && e.getData(CommonDataKeys.PROJECT) != null
    val project = e.getData(CommonDataKeys.PROJECT) ?: return
    val combinedState = project.service<CombinedState>()

    val muted = mutableListOf<String>()
    if (combinedState.state.settings.notificationsMuted.contains("completionResponseTimeIssues")) {
      muted += listOf(IssueName.SLOW_COMPLETION_RESPONSE_TIME, IssueName.HIGH_COMPLETION_TIMEOUT_RATE)
    }
    e.presentation.isVisible = combinedState.state.agentIssue != null && combinedState.state.agentIssue !in muted
    e.presentation.icon = if (combinedState.state.agentIssue == IssueName.CONNECTION_FAILED) {
      AllIcons.General.Error
    } else {
      AllIcons.General.Warning
    }
  }

  override fun getActionUpdateThread(): ActionUpdateThread {
    return ActionUpdateThread.BGT
  }
}