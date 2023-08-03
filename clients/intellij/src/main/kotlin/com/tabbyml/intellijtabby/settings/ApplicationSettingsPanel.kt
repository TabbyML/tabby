package com.tabbyml.intellijtabby.settings

import com.intellij.ui.components.JBCheckBox
import com.intellij.ui.components.JBTextField
import com.intellij.util.ui.FormBuilder
import javax.swing.JPanel

class ApplicationSettingsPanel {
  private val isAutoCompletionEnabledCheckBox = JBCheckBox("Enable auto completion")
  private val serverEndpointTextField = JBTextField()
  private val isAnonymousUsageTrackingDisabledCheckBox = JBCheckBox("Disable anonymous usage tracking")

  val mainPanel: JPanel = FormBuilder.createFormBuilder()
    .addLabeledComponent("Server endpoint", serverEndpointTextField, 1, false)
    .addTooltip("A http or https URL of Tabby server endpoint.")
    .addTooltip("If leave empty, server endpoint config in `~/.tabby/agent/config.toml` will be used")
    .addTooltip("Default to 'http://localhost:8080'.")
    .addSeparator()
    .addComponent(isAutoCompletionEnabledCheckBox, 1)
    .addComponent(isAnonymousUsageTrackingDisabledCheckBox, 1)
    .addComponentFillVertically(JPanel(), 0)
    .panel

  var isAutoCompletionEnabled: Boolean
    get() = isAutoCompletionEnabledCheckBox.isSelected
    set(value) {
      isAutoCompletionEnabledCheckBox.isSelected = value
    }

  var serverEndpoint: String
    get() = serverEndpointTextField.text
    set(value) {
      serverEndpointTextField.text = value
    }

  var isAnonymousUsageTrackingDisabled: Boolean
    get() = isAnonymousUsageTrackingDisabledCheckBox.isSelected
    set(value) {
      isAnonymousUsageTrackingDisabledCheckBox.isSelected = value
    }
}