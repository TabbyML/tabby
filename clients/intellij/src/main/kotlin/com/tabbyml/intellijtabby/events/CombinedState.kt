package com.tabbyml.intellijtabby.events

import com.intellij.openapi.Disposable
import com.intellij.openapi.components.Service
import com.intellij.openapi.components.service
import com.intellij.openapi.project.Project
import com.intellij.util.messages.Topic
import com.tabbyml.intellijtabby.lsp.ConnectionService
import com.tabbyml.intellijtabby.lsp.LanguageClient
import com.tabbyml.intellijtabby.lsp.protocol.Config
import com.tabbyml.intellijtabby.lsp.protocol.StatusInfo
import com.tabbyml.intellijtabby.notifications.hideAuthRequiredNotification
import com.tabbyml.intellijtabby.notifications.notifyAuthRequired
import com.tabbyml.intellijtabby.safeSyncPublisher
import com.tabbyml.intellijtabby.settings.SettingsService

@Service(Service.Level.PROJECT)
class CombinedState(private val project: Project) : Disposable {
  private val messageBusConnection = project.messageBus.connect()

  data class State(
    val settings: SettingsService.Settings,
    val connectionState: ConnectionService.State,
    val agentStatus: StatusInfo?,
    val agentConfig: Config?,
  ) {
    fun withSettings(settings: SettingsService.Settings): State {
      return State(settings, connectionState, agentStatus, agentConfig)
    }

    fun withConnectionState(connectionState: ConnectionService.State): State {
      return State(settings, connectionState, agentStatus, agentConfig)
    }

    fun withStatus(status: StatusInfo): State {
      return State(settings, connectionState, status, agentConfig)
    }

    fun withConfig(config: Config): State {
      return State(settings, connectionState, agentStatus, config)
    }
  }

  var state = State(
    service<SettingsService>().settings(),
    ConnectionService.State.INITIALIZING,
    null,
    null,
  )
    private set

  init {
    messageBusConnection.subscribe(SettingsService.Listener.TOPIC, object : SettingsService.Listener {
      override fun settingsChanged(settings: SettingsService.Settings) {
        state = state.withSettings(settings)
        project.safeSyncPublisher(Listener.TOPIC)?.stateChanged(state)
      }
    })
    messageBusConnection.subscribe(ConnectionService.Listener.TOPIC, object : ConnectionService.Listener {
      override fun connectionStateChanged(state: ConnectionService.State) {
        this@CombinedState.state = this@CombinedState.state.withConnectionState(state)
        project.safeSyncPublisher(Listener.TOPIC)?.stateChanged(this@CombinedState.state)
      }
    })
    messageBusConnection.subscribe(LanguageClient.StatusListener.TOPIC, object : LanguageClient.StatusListener {
      override fun statusChanged(status: StatusInfo) {
        state = state.withStatus(status)
        project.safeSyncPublisher(Listener.TOPIC)?.stateChanged(state)

        if (status.status == StatusInfo.Status.UNAUTHORIZED) {
          notifyAuthRequired()
        } else {
          hideAuthRequiredNotification()
        }
      }
    })
    messageBusConnection.subscribe(LanguageClient.ConfigListener.TOPIC, object : LanguageClient.ConfigListener {
      override fun configChanged(config: Config) {
        state = state.withConfig(config)
        project.safeSyncPublisher(Listener.TOPIC)?.stateChanged(state)
      }
    })
  }

  override fun dispose() {
    messageBusConnection.dispose()
  }

  interface Listener {
    fun stateChanged(state: State) {}

    companion object {
      @Topic.ProjectLevel
      val TOPIC = Topic(Listener::class.java, Topic.BroadcastDirection.NONE)
    }
  }
}