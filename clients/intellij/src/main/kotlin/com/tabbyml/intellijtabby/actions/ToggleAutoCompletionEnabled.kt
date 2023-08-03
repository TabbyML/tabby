package com.tabbyml.intellijtabby.actions

import com.intellij.openapi.actionSystem.ActionUpdateThread
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.components.service
import com.tabbyml.intellijtabby.settings.ApplicationSettingsState

class ToggleAutoCompletionEnabled : AnAction() {
  override fun actionPerformed(e: AnActionEvent) {
    val settings = service<ApplicationSettingsState>()
    settings.isAutoCompletionEnabled = !settings.isAutoCompletionEnabled
  }

  override fun update(e: AnActionEvent) {
    val settings = service<ApplicationSettingsState>()
    if (settings.isAutoCompletionEnabled) {
      e.presentation.text = "Disable Auto Completion"
      e.presentation.description = "Tabby does not show completion suggestions automatically, you can still request them on demand."
    } else {
      e.presentation.text = "Enable Auto Completion"
      e.presentation.description = "Tabby shows inline completion suggestions automatically."
    }
  }

  override fun getActionUpdateThread(): ActionUpdateThread {
    return ActionUpdateThread.BGT
  }
}