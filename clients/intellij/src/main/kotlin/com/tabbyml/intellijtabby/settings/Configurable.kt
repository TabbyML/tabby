package com.tabbyml.intellijtabby.settings

import com.intellij.openapi.components.service
import com.intellij.openapi.components.serviceOrNull
import com.intellij.openapi.options.Configurable
import com.intellij.openapi.project.Project
import javax.swing.JComponent

class Configurable(private val project: Project) : Configurable {
  private val settings = service<SettingsService>()
  private var settingsPanel: SettingsPanel? = null
  private val keymapSettings = project.serviceOrNull<KeymapSettings>()

  override fun getDisplayName(): String {
    return "Tabby"
  }

  override fun createComponent(): JComponent {
    val panel = SettingsPanel(project)
    settingsPanel = panel
    return panel.mainPanel
  }

  override fun isModified(): Boolean {
    val panel = settingsPanel ?: return false
    return panel.completionTriggerMode != settings.completionTriggerMode ||
        panel.serverEndpoint != settings.serverEndpoint ||
        panel.serverToken != settings.serverToken ||
        panel.nodeBinary != settings.nodeBinary ||
        panel.isAnonymousUsageTrackingDisabled != settings.isAnonymousUsageTrackingDisabled ||
        (panel.keymapStyle != keymapSettings?.getCurrentKeymapStyle() && panel.keymapStyle != KeymapSettings.KeymapStyle.CUSTOMIZE)
  }

  override fun apply() {
    val panel = settingsPanel ?: return
    settings.completionTriggerMode = panel.completionTriggerMode
    settings.serverEndpoint = panel.serverEndpoint
    settings.serverToken = panel.serverToken
    settings.nodeBinary = panel.nodeBinary
    settings.isAnonymousUsageTrackingDisabled = panel.isAnonymousUsageTrackingDisabled
    settings.notifyChanges(project)

    keymapSettings?.applyKeymapStyle(panel.keymapStyle)
  }

  override fun reset() {
    val panel = settingsPanel ?: return
    panel.completionTriggerMode = settings.completionTriggerMode
    panel.serverEndpoint = settings.serverEndpoint
    panel.serverToken = settings.serverToken
    panel.nodeBinary = settings.nodeBinary
    panel.isAnonymousUsageTrackingDisabled = settings.isAnonymousUsageTrackingDisabled

    keymapSettings?.let { panel.keymapStyle = it.getCurrentKeymapStyle() }
  }

  override fun disposeUIResources() {
    settingsPanel = null
  }
}