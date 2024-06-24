package com.tabbyml.intellijtabby.actions

import com.intellij.openapi.actionSystem.ActionUpdateThread
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.actionSystem.CommonDataKeys
import com.intellij.openapi.components.service
import com.tabbyml.intellijtabby.settings.SettingsState

class ToggleInlineCompletionTriggerMode : AnAction() {

  override fun actionPerformed(e: AnActionEvent) {
    val settings = e.getRequiredData(CommonDataKeys.PROJECT).service<SettingsState>()
    settings.completionTriggerMode = when (settings.completionTriggerMode) {
      SettingsState.TriggerMode.AUTOMATIC -> SettingsState.TriggerMode.MANUAL
      SettingsState.TriggerMode.MANUAL -> SettingsState.TriggerMode.AUTOMATIC
    }
  }

  override fun update(e: AnActionEvent) {
    e.presentation.isEnabled = e.project != null && e.getData(CommonDataKeys.PROJECT) != null
    val project = e.getData(CommonDataKeys.PROJECT) ?: return
    val settings = project.service<SettingsState>()
    if (settings.completionTriggerMode == SettingsState.TriggerMode.AUTOMATIC) {
      e.presentation.text = "Switch to Manual Mode"
      e.presentation.description = "Manual trigger inline completion suggestions on demand."
    } else {
      e.presentation.text = "Switch to Automatic Mode"
      e.presentation.description = "Show inline completion suggestions automatically."
    }
  }

  override fun getActionUpdateThread(): ActionUpdateThread {
    return ActionUpdateThread.BGT
  }
}