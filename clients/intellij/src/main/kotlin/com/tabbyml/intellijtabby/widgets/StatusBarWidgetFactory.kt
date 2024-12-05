package com.tabbyml.intellijtabby.widgets

import com.intellij.icons.AllIcons
import com.intellij.openapi.actionSystem.ActionGroup
import com.intellij.openapi.actionSystem.ActionManager
import com.intellij.openapi.actionSystem.DataContext
import com.intellij.openapi.application.invokeLater
import com.intellij.openapi.components.serviceOrNull
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
import com.tabbyml.intellijtabby.lsp.protocol.StatusInfo
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
        project.serviceOrNull<CombinedState>()?.state?.let { updateRendering(it) }
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
            val statusInfo = combinedState.agentStatus
            if (statusInfo == null) {
              icon = AnimatedIcon.Default()
              tooltip = "Tabby: Updating status"
            } else {
              icon = when (statusInfo.status) {
                StatusInfo.Status.CONNECTING, StatusInfo.Status.FETCHING -> {
                  AnimatedIcon.Default()
                }

                StatusInfo.Status.UNAUTHORIZED, StatusInfo.Status.COMPLETION_RESPONSE_SLOW -> {
                  AllIcons.General.Warning
                }

                StatusInfo.Status.DISCONNECTED -> {
                  AllIcons.General.Error
                }

                StatusInfo.Status.READY, StatusInfo.Status.READY_FOR_AUTO_TRIGGER -> {
                  AllIcons.Actions.Checked
                }

                StatusInfo.Status.READY_FOR_MANUAL_TRIGGER -> {
                  AllIcons.General.ChevronRight
                }

                else -> {
                  AnimatedIcon.Default()
                }
              }
              tooltip = statusInfo.tooltip ?: "Tabby: ${statusInfo.status}"
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
