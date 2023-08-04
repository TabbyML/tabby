package com.tabbyml.intellijtabby.settings

import com.intellij.openapi.components.service
import com.intellij.openapi.options.Configurable
import javax.swing.JComponent

class ApplicationConfigurable : Configurable {
  private lateinit var settingsPanel: ApplicationSettingsPanel

  override fun getDisplayName(): String {
    return "Tabby"
  }

  override fun createComponent(): JComponent {
    settingsPanel = ApplicationSettingsPanel()
    return settingsPanel.mainPanel
  }

  override fun isModified(): Boolean {
    val settings = service<ApplicationSettingsState>()
    return settingsPanel.isAutoCompletionEnabled != settings.isAutoCompletionEnabled
        || settingsPanel.serverEndpoint != settings.serverEndpoint
        || settingsPanel.isAnonymousUsageTrackingDisabled != settings.isAnonymousUsageTrackingDisabled
  }

  override fun apply() {
    val settings = service<ApplicationSettingsState>()
    settings.isAutoCompletionEnabled = settingsPanel.isAutoCompletionEnabled
    settings.serverEndpoint = settingsPanel.serverEndpoint
    settings.isAnonymousUsageTrackingDisabled = settingsPanel.isAnonymousUsageTrackingDisabled
  }

  override fun reset() {
    val settings = service<ApplicationSettingsState>()
    settingsPanel.isAutoCompletionEnabled = settings.isAutoCompletionEnabled
    settingsPanel.serverEndpoint = settings.serverEndpoint
    settingsPanel.isAnonymousUsageTrackingDisabled = settings.isAnonymousUsageTrackingDisabled
  }
}