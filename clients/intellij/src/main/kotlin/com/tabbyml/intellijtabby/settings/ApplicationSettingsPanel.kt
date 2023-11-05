package com.tabbyml.intellijtabby.settings

import com.intellij.ui.components.JBCheckBox
import com.intellij.ui.components.JBLabel
import com.intellij.ui.components.JBRadioButton
import com.intellij.ui.components.JBTextField
import com.intellij.util.ui.FormBuilder
import com.intellij.util.ui.JBUI
import com.intellij.util.ui.UIUtil
import javax.swing.ButtonGroup
import javax.swing.JPanel

private fun FormBuilder.addCopyableTooltip(text: String): FormBuilder {
  return this.addComponentToRightColumn(
    JBLabel(
      text,
      UIUtil.ComponentStyle.SMALL,
      UIUtil.FontColor.BRIGHTER
    ).apply {
      setBorder(JBUI.Borders.emptyLeft(10))
      setCopyable(true)
    },
    1,
  )
}

class ApplicationSettingsPanel {
  private val serverEndpointTextField = JBTextField()
  private val serverEndpointPanel = FormBuilder.createFormBuilder()
    .addComponent(serverEndpointTextField)
    .addCopyableTooltip(
      """
      <html>
      A http or https URL of Tabby server endpoint.<br/>
      If leave empty, server endpoint config in <i>~/.tabby-client/agent/config.toml</i> will be used.<br/>
      Default to <i>http://localhost:8080</i>.
      </html>
      """.trimIndent()
    )
    .panel

  private val nodeBinaryTextField = JBTextField()
  private val nodeBinaryPanel = FormBuilder.createFormBuilder()
    .addComponent(nodeBinaryTextField)
    .addCopyableTooltip(
      """
      <html>
      Path to the Node binary for running the Tabby agent. The Node version must be >= 18.0.<br/>
      If left empty, Tabby will attempt to find the Node binary in the <i>PATH</i> environment variable.<br/>
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
    .addCopyableTooltip("Trigger automatically when you stop typing")
    .addComponent(completionTriggerModeManualRadioButton)
    .addCopyableTooltip("Trigger manually by pressing `Alt + \\`")
    .panel

  private val isAnonymousUsageTrackingDisabledCheckBox = JBCheckBox("Disable anonymous usage tracking")
  private val isAnonymousUsageTrackingPanel: JPanel = FormBuilder.createFormBuilder()
    .addComponent(isAnonymousUsageTrackingDisabledCheckBox)
    .addCopyableTooltip(
      """
      <html>
      Tabby collects anonymous usage data and sends it to the Tabby team to help improve our products.<br/>
      Your code, generated completions, or any sensitive information is never tracked or sent.<br/>
      For more details on data collection, please check our <a href="https://tabby.tabbyml.com/docs/extensions/configuration#usage-collection">online documentation</a>.<br/>
      </html>
      """
    )
    .panel

  val mainPanel: JPanel = FormBuilder.createFormBuilder()
    .addLabeledComponent("Server endpoint", serverEndpointPanel, 5, false)
    .addSeparator(5)
    .addLabeledComponent("Inline completion trigger", completionTriggerModePanel, 5, false)
    .addSeparator(5)
    .addLabeledComponent("<html>Node binary<br/>(Requires restart IDE)</html>", nodeBinaryPanel, 5, false)
    .addSeparator(5)
    .addLabeledComponent("Anonymous usage tracking", isAnonymousUsageTrackingPanel, 5, false)
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

  var nodeBinary: String
    get() = nodeBinaryTextField.text
    set(value) {
      nodeBinaryTextField.text = value
    }

  var isAnonymousUsageTrackingDisabled: Boolean
    get() = isAnonymousUsageTrackingDisabledCheckBox.isSelected
    set(value) {
      isAnonymousUsageTrackingDisabledCheckBox.isSelected = value
    }
}