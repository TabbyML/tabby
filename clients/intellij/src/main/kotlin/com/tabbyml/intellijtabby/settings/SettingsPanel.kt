package com.tabbyml.intellijtabby.settings

import com.intellij.openapi.application.ModalityState
import com.intellij.openapi.application.invokeLater
import com.intellij.openapi.components.service
import com.intellij.openapi.components.serviceOrNull
import com.intellij.openapi.keymap.impl.ui.KeymapPanel
import com.intellij.openapi.options.ShowSettingsUtil
import com.intellij.openapi.progress.ProgressIndicator
import com.intellij.openapi.progress.ProgressManager
import com.intellij.openapi.progress.Task
import com.intellij.openapi.project.Project
import com.intellij.openapi.ui.Messages
import com.intellij.ui.components.*
import com.intellij.util.ui.FormBuilder
import com.intellij.util.ui.JBUI
import com.intellij.util.ui.UIUtil
import com.tabbyml.intellijtabby.lsp.ConnectionService
import com.tabbyml.intellijtabby.lsp.protocol.StatusIgnoredIssuesEditParams
import com.tabbyml.intellijtabby.lsp.protocol.StatusInfo
import com.tabbyml.intellijtabby.lsp.protocol.StatusRequestParams
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.future.await
import kotlinx.coroutines.launch
import javax.swing.ButtonGroup
import javax.swing.JButton
import javax.swing.JPanel

class SettingsPanel(private val project: Project) {
  private val settings = service<SettingsService>()
  private val scope = CoroutineScope(Dispatchers.IO)
  private suspend fun getServer() = project.serviceOrNull<ConnectionService>()?.getServerAsync()

  private val serverEndpointTextField = JBTextField()
  private val serverTokenPasswordField = JBPasswordField()
  private val serverEndpointCheckConnectionButton = JButton("Check connection").apply {
    addActionListener {
      val parentComponent = this@SettingsPanel.mainPanel

      val task = object : Task.Modal(
        project, parentComponent, "Check Connection", true
      ) {
        lateinit var job: Job
        override fun run(indicator: ProgressIndicator) {
          job = scope.launch {
            indicator.isIndeterminate = true
            indicator.text = "Checking connection..."
            settings.serverEndpoint = serverEndpoint
            settings.serverToken = serverToken
            settings.notifyChanges(project)

            val server = getServer() ?: return@launch
            val statusInfo = server.statusFeature.getStatus(StatusRequestParams(recheckConnection = true)).await()
            when (statusInfo.status) {
              StatusInfo.Status.CONNECTING -> {
                // Do nothing
              }

              StatusInfo.Status.UNAUTHORIZED -> {
                invokeLater(ModalityState.stateForComponent(parentComponent)) {
                  Messages.showErrorDialog(
                    parentComponent,
                    "Tabby server requires authentication, please set your personal token.",
                    "Check Connection Failed"
                  )
                }
              }

              StatusInfo.Status.DISCONNECTED -> {
                invokeLater(ModalityState.stateForComponent(parentComponent)) {
                  val errorMessage = statusInfo.helpMessage ?: "Unknown error."
                  val messages = "<html>Failed to connect to the Tabby server:<br/>${errorMessage}</html>"
                  Messages.showErrorDialog(parentComponent, messages, "Check Connection Failed")
                }
              }

              else -> {
                invokeLater(ModalityState.stateForComponent(parentComponent)) {
                  Messages.showInfoMessage(
                    parentComponent, "Successfully connected to the Tabby server.", "Check Connection Completed"
                  )
                }
              }
            }
          }

          while (job.isActive) {
            indicator.checkCanceled()
            Thread.sleep(100)
          }
        }

        override fun onCancel() {
          job.cancel()
        }
      }

      ProgressManager.getInstance().run(task)
    }
  }
  private val serverEndpointPanel =
    FormBuilder.createFormBuilder().addLabeledComponent("Endpoint", serverEndpointTextField, 0, false)
      .addCopyableTooltip(
        """
      <html>
      A http or https URL of Tabby server endpoint.<br/>
      If left empty, the server endpoint config in <i>~/.tabby-client/agent/config.toml</i> will be used.<br/>
      Default to <i>http://localhost:8080</i>.
      </html>
      """.trimIndent()
      ).addSeparator().addLabeledComponent("Token", serverTokenPasswordField, 0, false).addCopyableTooltip(
        """
      <html>
      Set token here if your Tabby server requires authentication.
      </html>
      """.trimIndent()
      ).addSeparator().addComponent(serverEndpointCheckConnectionButton).panel

  private val nodeBinaryTextField = JBTextField()
  private val nodeBinaryPanel = FormBuilder.createFormBuilder().addComponent(nodeBinaryTextField).addCopyableTooltip(
    """
      <html>
      Path to the Node binary for running the Tabby agent. The Node version must be >= 18.0.<br/>
      If left empty, Tabby will attempt to find the Node binary in the <i>PATH</i> environment variable.<br/>
      </html>
      """.trimIndent()
  ).panel

  private val completionTriggerModeAutomaticRadioButton = JBRadioButton("Automatic")
  private val completionTriggerModeManualRadioButton = JBRadioButton("Manual")
  private val completionTriggerModeRadioGroup = ButtonGroup().apply {
    add(completionTriggerModeAutomaticRadioButton)
    add(completionTriggerModeManualRadioButton)
  }
  private val completionTriggerModePanel: JPanel =
    FormBuilder.createFormBuilder().addComponent(completionTriggerModeAutomaticRadioButton)
      .addCopyableTooltip("Trigger automatically when you stop typing")
      .addComponent(completionTriggerModeManualRadioButton)
      .addCopyableTooltip("Trigger on-demand by pressing a shortcut").panel

  private val keymapStyleDefaultRadioButton = JBRadioButton("Default")
  private val keymapStyleTabbyStyleRadioButton = JBRadioButton("Tabby style")
  private val keymapStyleCustomRadioButton = JBRadioButton("<html><a href=''>Customize...</a><html>").apply {
    addActionListener {
      ShowSettingsUtil.getInstance().showSettingsDialog(project, KeymapPanel::class.java) { panel ->
        CoroutineScope(Dispatchers.IO).launch {
          Thread.sleep(500) // FIXME: It seems that we need to wait for the KeymapPanel to be ready?
          invokeLater(ModalityState.stateForComponent(panel)) {
            panel.showOption("Tabby")
          }
        }
      }
    }
    border = JBUI.Borders.emptyLeft(1)
  }
  private val keymapStyleRadioGroup = ButtonGroup().apply {
    add(keymapStyleDefaultRadioButton)
    add(keymapStyleTabbyStyleRadioButton)
    add(keymapStyleCustomRadioButton)
  }
  private val keymapStylePanel: JPanel = FormBuilder.createFormBuilder().addComponent(keymapStyleDefaultRadioButton)
    .addCopyableTooltip("<html>Use <i>Tab</i> to accept full completion, and use <i>Ctrl+Tab</i> to accept next line.</html>")
    .addComponent(keymapStyleTabbyStyleRadioButton)
    .addCopyableTooltip("<html>Use <i>Ctrl+Tab</i> to accept full completion, and use <i>Tab</i> to accept next line.</html>")
    .addComponent(keymapStyleCustomRadioButton).panel

  private val isAnonymousUsageTrackingDisabledCheckBox = JBCheckBox("Disable anonymous usage tracking")
  private val isAnonymousUsageTrackingPanel: JPanel =
    FormBuilder.createFormBuilder().addComponent(isAnonymousUsageTrackingDisabledCheckBox).addCopyableTooltip(
      """
      <html>
      Tabby collects aggregated anonymous usage data and sends it to the Tabby team to help improve our products.<br/>
      Your code, generated completions, or any identifying information is never tracked or transmitted.<br/>
      For more details on data collection, please check our <a href="https://tabby.tabbyml.com/docs/extensions/configurations#usage-collection">online documentation</a>.<br/>
      </html>
      """
    ).panel

  private val resetMutedNotificationsButton = JButton("Reset \"Don't Show Again\" Notifications").apply {
    addActionListener {
      scope.launch {
        val server = getServer() ?: return@launch
        server.statusFeature.editIgnoredIssues(StatusIgnoredIssuesEditParams(operation = StatusIgnoredIssuesEditParams.Operation.REMOVE_ALL))
          .thenAccept {
            invokeLater(ModalityState.stateForComponent(this@SettingsPanel.mainPanel)) {
              Messages.showInfoMessage("Reset \"Don't Show Again\" notifications successfully.", "Reset Notifications")
            }
          }
      }
    }
  }
  private val resetMutedNotificationsPanel: JPanel =
    FormBuilder.createFormBuilder().addComponent(resetMutedNotificationsButton).panel

  val mainPanel: JPanel =
    FormBuilder.createFormBuilder().addLabeledComponent("Server", serverEndpointPanel, 5, false)
      .addSeparator(5)
      .addLabeledComponent("Inline completion trigger", completionTriggerModePanel, 5, false)
      .addSeparator(5)
      .addLabeledComponent("Keymap", keymapStylePanel, 5, false)
      .addSeparator(5)
      .addLabeledComponent("<html>Node binary<br/>(Requires restart IDE)</html>", nodeBinaryPanel, 5, false)
      .addSeparator(5)
      .addLabeledComponent("Anonymous usage tracking", isAnonymousUsageTrackingPanel, 5, false)
      .addSeparator(5)
      .addLabeledComponent("Notifications", resetMutedNotificationsPanel, 5, false)
      .addComponentFillVertically(JPanel(), 0)
      .panel

  var serverEndpoint: String
    get() = serverEndpointTextField.text
    set(value) {
      serverEndpointTextField.text = value
    }

  var serverToken: String
    get() = String(serverTokenPasswordField.password)
    set(value) {
      serverTokenPasswordField.text = value
    }

  var nodeBinary: String
    get() = nodeBinaryTextField.text
    set(value) {
      nodeBinaryTextField.text = value
    }

  var keymapStyle: KeymapSettings.KeymapStyle
    get() = if (keymapStyleDefaultRadioButton.isSelected) {
      KeymapSettings.KeymapStyle.DEFAULT
    } else if (keymapStyleTabbyStyleRadioButton.isSelected) {
      KeymapSettings.KeymapStyle.TABBY_STYLE
    } else {
      KeymapSettings.KeymapStyle.CUSTOMIZE
    }
    set(value) {
      when (value) {
        KeymapSettings.KeymapStyle.DEFAULT -> keymapStyleDefaultRadioButton.isSelected = true
        KeymapSettings.KeymapStyle.TABBY_STYLE -> keymapStyleTabbyStyleRadioButton.isSelected = true
        KeymapSettings.KeymapStyle.CUSTOMIZE -> keymapStyleCustomRadioButton.isSelected = true
      }
    }

  var completionTriggerMode: SettingsState.TriggerMode
    get() = if (completionTriggerModeAutomaticRadioButton.isSelected) {
      SettingsState.TriggerMode.AUTOMATIC
    } else {
      SettingsState.TriggerMode.MANUAL
    }
    set(value) {
      when (value) {
        SettingsState.TriggerMode.AUTOMATIC -> completionTriggerModeAutomaticRadioButton.isSelected = true
        SettingsState.TriggerMode.MANUAL -> completionTriggerModeManualRadioButton.isSelected = true
      }
    }

  var isAnonymousUsageTrackingDisabled: Boolean
    get() = isAnonymousUsageTrackingDisabledCheckBox.isSelected
    set(value) {
      isAnonymousUsageTrackingDisabledCheckBox.isSelected = value
    }
}

private fun FormBuilder.addCopyableTooltip(text: String): FormBuilder {
  return this.addComponentToRightColumn(
    JBLabel(
      text, UIUtil.ComponentStyle.SMALL, UIUtil.FontColor.BRIGHTER
    ).apply {
      setBorder(JBUI.Borders.emptyLeft(10))
      setCopyable(true)
    },
    1,
  )
}
