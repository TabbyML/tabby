package com.tabbyml.intellijtabby.status

import com.intellij.icons.AllIcons
import com.intellij.openapi.actionSystem.*
import com.intellij.openapi.application.invokeLater
import com.intellij.openapi.components.service
import com.intellij.openapi.project.Project
import com.intellij.openapi.ui.popup.JBPopupFactory
import com.intellij.openapi.ui.popup.ListPopup
import com.intellij.openapi.vfs.VirtualFile
import com.intellij.openapi.wm.StatusBarWidget
import com.intellij.openapi.wm.impl.status.EditorBasedStatusBarPopup
import com.intellij.openapi.wm.impl.status.widget.StatusBarEditorBasedWidgetFactory
import com.tabbyml.intellijtabby.agent.Agent
import com.tabbyml.intellijtabby.agent.AgentService
import com.tabbyml.intellijtabby.settings.ApplicationSettingsState
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.launch

class StatusBarWidgetFactory : StatusBarEditorBasedWidgetFactory() {
  override fun getId(): String {
    return StatusBarWidgetFactory::class.java.name
  }

  override fun getDisplayName(): String {
    return "Tabby"
  }

  override fun createWidget(project: Project): StatusBarWidget {
    return object : EditorBasedStatusBarPopup(project, false) {
      val scope: CoroutineScope = CoroutineScope(Dispatchers.Main)
      val text = "Tabby"
      var icon = AllIcons.Actions.Refresh
      var tooltip = "Tabby: Initializing"

      init {
        val settings = service<ApplicationSettingsState>()
        val agentService = service<AgentService>()
        scope.launch {
          settings.state.combine(agentService.status) { settings, agentStatus ->
            Pair(settings, agentStatus)
          }.collect {
            updateStatus(it.first, it.second)
          }
        }
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

      override fun createPopup(context: DataContext?): ListPopup? {
        if (context == null) {
          return null
        }
        return JBPopupFactory.getInstance().createActionGroupPopup(
          tooltip,
          object : ActionGroup() {
            override fun getChildren(e: AnActionEvent?): Array<AnAction> {
              val actionManager = ActionManager.getInstance()
              return arrayOf(
                actionManager.getAction("Tabby.ToggleAutoCompletionEnabled"),
                actionManager.getAction("Tabby.OpenSettings"),
              )
            }
          },
          context,
          false,
          null,
          10,
        )
      }

      private fun updateStatus(settingsState: ApplicationSettingsState.State, agentStatus: Agent.Status) {
        if (!settingsState.isAutoCompletionEnabled) {
          icon = AllIcons.Windows.CloseSmall
          tooltip = "Tabby: Auto completion is disabled"
        } else {
          when(agentStatus) {
            Agent.Status.NOT_INITIALIZED -> {
              icon = AllIcons.Actions.Refresh
              tooltip = "Tabby: Initializing"
            }
            Agent.Status.READY -> {
              icon = AllIcons.Actions.Checked
              tooltip = "Tabby: Ready"
            }
            Agent.Status.DISCONNECTED -> {
              icon = AllIcons.General.Error
              tooltip = "Tabby: Cannot connect to Server"
            }
            Agent.Status.UNAUTHORIZED -> {
              icon = AllIcons.General.Error
              tooltip = "Tabby: Requires authorization"
            }
          }
        }
        invokeLater {
          update { myStatusBar?.updateWidget(ID()) }
        }
      }
    }
  }

  override fun disposeWidget(widget: StatusBarWidget) {
    // Nothing to do
  }
}
