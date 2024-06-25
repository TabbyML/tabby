package com.tabbyml.intellijtabby.widgets

import com.intellij.icons.AllIcons
import com.intellij.openapi.actionSystem.ActionGroup
import com.intellij.openapi.actionSystem.ActionManager
import com.intellij.openapi.actionSystem.DataContext
import com.intellij.openapi.application.invokeLater
import com.intellij.openapi.components.service
import com.intellij.openapi.project.Project
import com.intellij.openapi.ui.popup.JBPopupFactory
import com.intellij.openapi.ui.popup.ListPopup
import com.intellij.openapi.vfs.VirtualFile
import com.intellij.openapi.wm.StatusBarWidget
import com.intellij.openapi.wm.impl.status.EditorBasedStatusBarPopup
import com.intellij.openapi.wm.impl.status.widget.StatusBarEditorBasedWidgetFactory
import com.intellij.ui.AnimatedIcon
import com.tabbyml.intellijtabby.events.CombinedState
import com.tabbyml.intellijtabby.lsp.ConnectionService
import com.tabbyml.intellijtabby.lsp.protocol.IssueName
import com.tabbyml.intellijtabby.lsp.protocol.Status
import com.tabbyml.intellijtabby.settings.SettingsState
import javax.swing.Icon

class StatusBarWidgetFactory : StatusBarEditorBasedWidgetFactory() {
  override fun getId(): String {
    return StatusBarWidgetFactory::class.java.name
  }

  override fun getDisplayName(): String {
    return "Tabby"
  }

  override fun createWidget(project: Project): StatusBarWidget {
    return object : EditorBasedStatusBarPopup(project, false) {
      private val messageBusConnection = project.messageBus.connect()
      val text = "Tabby"
      var icon: Icon = AnimatedIcon.Default()
      var tooltip = "Tabby: Initializing"

      init {
        updateRendering(project.service<CombinedState>().state)
        messageBusConnection.subscribe(CombinedState.Listener.TOPIC, object : CombinedState.Listener {
          override fun stateChanged(state: CombinedState.State) {
            updateRendering(state)
          }
        })
      }

      override fun ID(): String {
        return "${StatusBarWidgetFactory::class.java.name}.widget"
      }

      override fun createInstance(project: Project): StatusBarWidget {
        return createWidget(project)
      }

      override fun getWidgetState(file: VirtualFile?): WidgetState {
        return WidgetState(tooltip, text, true).also {
          it.icon = icon
        }
      }

      override fun createPopup(context: DataContext): ListPopup {
        return JBPopupFactory.getInstance().createActionGroupPopup(
          tooltip,
          ActionManager.getInstance().getAction("Tabby.StatusBarPopupMenu") as ActionGroup,
          context,
          false,
          null,
          10,
        )
      }

      override fun dispose() {
        messageBusConnection.dispose()
        super.dispose()
      }

      private fun updateRendering(combinedState: CombinedState.State) {
        when (combinedState.connectionState) {
          ConnectionService.State.INITIALIZING -> {
            icon = AnimatedIcon.Default()
            tooltip = "Tabby: Initializing"
          }

          ConnectionService.State.INITIALIZATION_FAILED -> {
            icon = AllIcons.General.Error
            tooltip = "Tabby: Initialization failed"
          }

          ConnectionService.State.READY -> {
            when (combinedState.agentStatus) {
              Status.NOT_INITIALIZED, Status.FINALIZED -> {
                icon = AnimatedIcon.Default()
                tooltip = "Tabby: Initializing"
              }

              Status.DISCONNECTED -> {
                icon = AllIcons.General.Error
                tooltip = "Tabby: Cannot connect to Server, please check settings"
              }

              Status.UNAUTHORIZED -> {
                icon = AllIcons.General.Warning
                tooltip = "Tabby: Authorization required, please set your personal token in settings"
              }

              Status.READY -> {
                val muted = mutableListOf<String>()
                if (combinedState.settings.notificationsMuted.contains("completionResponseTimeIssues")) {
                  muted += listOf(IssueName.SLOW_COMPLETION_RESPONSE_TIME, IssueName.HIGH_COMPLETION_TIMEOUT_RATE)
                }
                val agentIssue = combinedState.agentIssue
                if (agentIssue != null && agentIssue !in muted) {
                  icon = AllIcons.General.Warning
                  tooltip = when (agentIssue) {
                    IssueName.SLOW_COMPLETION_RESPONSE_TIME -> "Tabby: Completion requests appear to take too much time"
                    IssueName.HIGH_COMPLETION_TIMEOUT_RATE -> "Tabby: Most completion requests timed out"
                    IssueName.CONNECTION_FAILED -> "Tabby: Cannot connect to Server, please check settings"
                    else -> "Tabby: Please check issues"
                  }
                } else if (combinedState.isInlineCompletionLoading) {
                  icon = AnimatedIcon.Default()
                  tooltip = "Tabby: Generating code completions"
                } else {
                  when (combinedState.settings.completionTriggerMode) {
                    SettingsState.TriggerMode.AUTOMATIC -> {
                      icon = AllIcons.Actions.Checked
                      tooltip = "Tabby: Automatic code completion is enabled"
                    }

                    SettingsState.TriggerMode.MANUAL -> {
                      icon = AllIcons.General.ChevronRight
                      tooltip = "Tabby: Standing by, please manually trigger code completion."
                    }
                  }
                }
              }
            }
          }
        }
        invokeLater {
          update { myStatusBar?.updateWidget(ID()) }
        }
      }
    }
  }
}
