package com.tabbyml.intellijtabby.settings

import com.intellij.ui.components.JBCheckBox
import com.intellij.ui.components.JBRadioButton
import com.intellij.ui.components.JBTextField
import com.intellij.util.ui.FormBuilder
import javax.swing.ButtonGroup
import javax.swing.JPanel

class ApplicationSettingsPanel {
  private val serverEndpointTextField = JBTextField()
  private val serverEndpointPanel = FormBuilder.createFormBuilder()
    .addComponent(serverEndpointTextField)
    .addTooltip(
      """
      <html>
      A http or https URL of Tabby server endpoint.<br/>
      If leave empty, server endpoint config in <i>~/.tabby-client/agent/config.toml</i> will be used<br/>
      Default to <i>http://localhost:8080</i>.
      </html>
      """.trimIndent()
    )
    .panel

  private val completionTriggerModeAutomaticRadioButton = JBRadioButton("Automatic")
  private val completionTriggerModeManualRadioButton = JBRadioButton("Manual")
  private val completionTriggerModeRadioGroup = ButtonGroup().apply {
    add(completionTriggerModeAutomaticRadioButton)
    add(completionTriggerModeManualRadioButton)
  }
  private val completionTriggerModePanel: JPanel = FormBuilder.createFormBuilder()
    .addComponent(completionTriggerModeAutomaticRadioButton)
    .addTooltip("Trigger automatically when you stop typing")
    .addComponent(completionTriggerModeManualRadioButton)
    .addTooltip("Trigger manually by pressing `Alt + \\`")
    .panel

  private val isAnonymousUsageTrackingDisabledCheckBox = JBCheckBox("Disable")

  val mainPanel: JPanel = FormBuilder.createFormBuilder()
    .addLabeledComponent("Server endpoint", serverEndpointPanel, 5, false)
    .addSeparator(5)
    .addLabeledComponent("Inline completion trigger", completionTriggerModePanel, 5, false)
    .addSeparator(5)
    .addLabeledComponent("Anonymous usage tracking", isAnonymousUsageTrackingDisabledCheckBox, 5, false)
    .addComponentFillVertically(JPanel(), 0)
    .panel

  var completionTriggerMode: ApplicationSettingsState.TriggerMode
    get() = if (completionTriggerModeAutomaticRadioButton.isSelected) {
      ApplicationSettingsState.TriggerMode.AUTOMATIC
    } else {
      ApplicationSettingsState.TriggerMode.MANUAL
    }
    set(value) {
      when (value) {
        ApplicationSettingsState.TriggerMode.AUTOMATIC -> completionTriggerModeAutomaticRadioButton.isSelected = true
        ApplicationSettingsState.TriggerMode.MANUAL -> completionTriggerModeManualRadioButton.isSelected = true
      }
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