package com.tabbyml.intellijtabby.actions

import com.intellij.openapi.actionSystem.ActionUpdateThread
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.components.service
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.progress.ProgressIndicator
import com.intellij.openapi.progress.ProgressManager
import com.intellij.openapi.progress.Task
import com.tabbyml.intellijtabby.agent.Agent
import com.tabbyml.intellijtabby.agent.AgentService
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

@Deprecated("Tabby Cloud auth support will be removed.")
class OpenAuthPage : AnAction() {
  private val logger = Logger.getInstance(OpenAuthPage::class.java)

  override fun actionPerformed(e: AnActionEvent) {
    val agentService = service<AgentService>()
    agentService.authNotification?.expire()

    val task = object : Task.Modal(
      e.project,
      "Tabby Server Authorization",
      true
    ) {
      lateinit var job: Job
      override fun run(indicator: ProgressIndicator) {
        job = agentService.scope.launch {
          agentService.requestAuth(indicator)
        }
        logger.info("Authorization task started.")
        while (job.isActive) {
          indicator.checkCanceled()
          Thread.sleep(100)
        }
      }

      override fun onCancel() {
        logger.info("Authorization task cancelled.")
        job.cancel()
      }
    }
    ProgressManager.getInstance().run(task)
  }

  override fun update(e: AnActionEvent) {
    val agentService = service<AgentService>()
    e.presentation.isVisible = agentService.status.value == Agent.Status.UNAUTHORIZED
  }

  override fun getActionUpdateThread(): ActionUpdateThread {
    return ActionUpdateThread.BGT
  }
}