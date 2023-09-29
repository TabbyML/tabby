package com.tabbyml.intellijtabby.actions

import com.intellij.ide.BrowserUtil
import com.intellij.openapi.actionSystem.ActionUpdateThread
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.application.invokeLater
import com.intellij.openapi.components.service
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.ui.Messages
import com.tabbyml.intellijtabby.agent.AgentService
import kotlinx.coroutines.launch

class CheckIssueDetail : AnAction() {
  private val logger = Logger.getInstance(CheckIssueDetail::class.java)

  override fun actionPerformed(e: AnActionEvent) {
    val agentService = service<AgentService>()
    agentService.issueNotification?.expire()

    agentService.scope.launch {
      val detail = agentService.getCurrentIssueDetail() ?: return@launch
      val serverHealthState = agentService.getServerHealthState()
      logger.info("Show issue detail: $detail, $serverHealthState")
      val title = when (detail["name"]) {
        "slowCompletionResponseTime" -> "Completion Requests Appear to Take Too Much Time"
        "highCompletionTimeoutRate" -> "Most Completion Requests Timed Out"
        else -> return@launch
      }
      val message = buildDetailMessage(detail, serverHealthState)
      invokeLater {
        val result = Messages.showOkCancelDialog(message, title, "Dismiss", "Supported Models", Messages.getInformationIcon())
        if (result == Messages.CANCEL) {
          BrowserUtil.browse("https://tabby.tabbyml.com/docs/models/")
        }
      }
    }
  }

  private fun buildDetailMessage(detail: Map<String, Any>, serverHealthState: Map<String, Any>?): String {
    val stats = detail["completionResponseStats"] as Map<*, *>?
    val statsMessages = when (detail["name"]) {
      "slowCompletionResponseTime" -> if (stats != null && stats["responses"] is Number && stats["averageResponseTime"] is Number) {
        val response = (stats["responses"] as Number).toInt()
        val averageResponseTime = (stats["averageResponseTime"] as Number).toInt()
        "The average response time of recent $response completion requests is $averageResponseTime ms.\n\n"
      } else {
        ""
      }

      "highCompletionTimeoutRate" -> if (stats != null && stats["total"] is Number && stats["timeouts"] is Number) {
        val timeout = (stats["timeouts"] as Number).toInt()
        val total = (stats["total"] as Number).toInt()
        "$timeout of $total completion requests timed out.\n\n"
      } else {
        ""
      }

      else -> ""
    }

    val device = serverHealthState?.get("device") as String? ?: ""
    val model = serverHealthState?.get("model") as String? ?: ""
    val helpMessageForRunningLargeModelOnCPU = if (device == "cpu" && model.endsWith("B")) {
      """
      Your Tabby server is running model $model on CPU.
      This model is too large to run on CPU, please try a smaller model or switch to GPU.
      You can find supported model list in online documents.
      """
    } else {
      ""
    }
    var helpMessage = ""
    if (helpMessageForRunningLargeModelOnCPU.isNotEmpty()) {
      helpMessage += helpMessageForRunningLargeModelOnCPU + "\n\n"
      helpMessage += "Other possible causes of this issue are: \n"
    } else {
      helpMessage += "Possible causes of this issue are: \n";
    }
    helpMessage += " - A poor network connection. Please check your network and proxy settings.\n";
    helpMessage += " - Server overload. Please contact your Tabby server administrator for assistance.\n";
    if (helpMessageForRunningLargeModelOnCPU.isEmpty()) {
      helpMessage += " - The running model $model is too large to run on your Tabby server. ";
      helpMessage += "Please try a smaller model. You can find supported model list in online documents.\n";
    }
    return statsMessages + helpMessage
  }

  override fun update(e: AnActionEvent) {
    val agentService = service<AgentService>()
    e.presentation.isVisible = agentService.currentIssue.value != null
  }

  override fun getActionUpdateThread(): ActionUpdateThread {
    return ActionUpdateThread.BGT
  }
}