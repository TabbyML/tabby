package com.tabbyml.intellijtabby.actions

import com.intellij.openapi.actionSystem.ActionUpdateThread
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.components.service
import com.tabbyml.intellijtabby.settings.ApplicationSettingsState

class ToggleInlineCompletionTriggerMode : AnAction() {
  override fun actionPerformed(e: AnActionEvent) {
    val settings = service<ApplicationSettingsState>()
    settings.completionTriggerMode = when (settings.completionTriggerMode) {
      ApplicationSettingsState.TriggerMode.AUTOMATIC -> ApplicationSettingsState.TriggerMode.MANUAL
      ApplicationSettingsState.TriggerMode.MANUAL -> ApplicationSettingsState.TriggerMode.AUTOMATIC
    }
  }

  override fun update(e: AnActionEvent) {
    val settings = service<ApplicationSettingsState>()
    if (settings.completionTriggerMode == ApplicationSettingsState.TriggerMode.AUTOMATIC) {
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