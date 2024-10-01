package com.tabbyml.intellijtabby.events

import com.intellij.openapi.Disposable
import com.intellij.openapi.components.Service
import com.intellij.openapi.components.service
import com.intellij.openapi.project.Project
import com.intellij.util.messages.Topic
import com.tabbyml.intellijtabby.completion.InlineCompletionService
import com.tabbyml.intellijtabby.lsp.ConnectionService
import com.tabbyml.intellijtabby.lsp.LanguageClient
import com.tabbyml.intellijtabby.lsp.protocol.IssueList
import com.tabbyml.intellijtabby.lsp.protocol.ServerInfo
import com.tabbyml.intellijtabby.lsp.protocol.Status
import com.tabbyml.intellijtabby.notifications.notifyAuthRequired
import com.tabbyml.intellijtabby.safeSyncPublisher
import com.tabbyml.intellijtabby.settings.SettingsService

@Service(Service.Level.PROJECT)
class CombinedState(private val project: Project) : Disposable {
  private val messageBusConnection = project.messageBus.connect()

  data class State(
    val settings: SettingsService.Settings,
    val connectionState: ConnectionService.State,
    val agentStatus: String,
    val agentIssue: String?,
    val agentServerInfo: ServerInfo?,
    val isInlineCompletionLoading: Boolean,
  ) {
    fun withSettings(settings: SettingsService.Settings): State {
      return State(settings, connectionState, agentStatus, agentIssue, agentServerInfo, isInlineCompletionLoading)
    }

    fun withConnectionState(connectionState: ConnectionService.State): State {
      return State(settings, connectionState, agentStatus, agentIssue, agentServerInfo, isInlineCompletionLoading)
    }

    fun withAgentStatus(agentStatus: String): State {
      return State(settings, connectionState, agentStatus, agentIssue, agentServerInfo, isInlineCompletionLoading)
    }

    fun withAgentIssue(currentIssue: String?): State {
      return State(settings, connectionState, agentStatus, currentIssue, agentServerInfo, isInlineCompletionLoading)
    }

    fun withoutAgentIssue(): State {
      return withAgentIssue(null)
    }

    fun withAgentServerInfo(serverInfo: ServerInfo?): State {
      return State(settings, connectionState, agentStatus, agentIssue, serverInfo, isInlineCompletionLoading)
    }

    fun withInlineCompletionLoading(isInlineCompletionLoading: Boolean = true): State {
      return State(settings, connectionState, agentStatus, agentIssue, agentServerInfo, isInlineCompletionLoading)
    }
  }

  var state = State(
    service<SettingsService>().settings(),
    ConnectionService.State.INITIALIZING,
    Status.NOT_INITIALIZED,
    null,
    null,
    false,
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
    messageBusConnection.subscribe(LanguageClient.AgentListener.TOPIC, object : LanguageClient.AgentListener {
      override fun agentStatusChanged(status: String) {
        state = state.withAgentStatus(status)
        project.safeSyncPublisher(Listener.TOPIC)?.stateChanged(state)

        if (status == Status.UNAUTHORIZED) {
          notifyAuthRequired()
        }
      }

      override fun agentIssueUpdated(issueList: IssueList) {
        state = issueList.issues.firstOrNull()?.let {
          state.withAgentIssue(it)
        } ?: state.withoutAgentIssue()
        project.safeSyncPublisher(Listener.TOPIC)?.stateChanged(state)
      }

      override fun agentServerInfoUpdated(serverInfo: ServerInfo) {
        state = state.withAgentServerInfo(serverInfo)
        project.safeSyncPublisher(Listener.TOPIC)?.stateChanged(state)
      }
    })

    messageBusConnection.subscribe(InlineCompletionService.Listener.TOPIC, object : InlineCompletionService.Listener {
      override fun loadingStateChanged(loading: Boolean) {
        state = state.withInlineCompletionLoading(loading)
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