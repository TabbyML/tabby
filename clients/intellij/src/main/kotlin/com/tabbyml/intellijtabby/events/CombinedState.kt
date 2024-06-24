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
import com.tabbyml.intellijtabby.lsp.protocol.IssueName
import com.tabbyml.intellijtabby.lsp.protocol.Status
import com.tabbyml.intellijtabby.settings.SettingsState

@Service(Service.Level.PROJECT)
class CombinedState(private val project: Project) : Disposable {

  private val messageBusConnection = project.messageBus.connect()
  private val publisher = project.messageBus.syncPublisher(Listener.TOPIC)

  data class State(
    val settings: SettingsState.Settings,
    val connectionState: ConnectionService.State,
    val agentStatus: Status,
    val agentIssue: IssueName?,
    val isInlineCompletionLoading: Boolean,
  ) {
    fun withSettings(settings: SettingsState.Settings): State {
      return State(settings, connectionState, agentStatus, agentIssue, isInlineCompletionLoading)
    }

    fun withConnectionState(connectionState: ConnectionService.State): State {
      return State(settings, connectionState, agentStatus, agentIssue, isInlineCompletionLoading)
    }

    fun withAgentStatus(agentStatus: Status): State {
      return State(settings, connectionState, agentStatus, agentIssue, isInlineCompletionLoading)
    }

    fun withAgentIssue(currentIssue: IssueName?): State {
      return State(settings, connectionState, agentStatus, currentIssue, isInlineCompletionLoading)
    }

    fun withoutAgentIssue(): State {
      return withAgentIssue(null)
    }

    fun withInlineCompletionLoading(isInlineCompletionLoading: Boolean = true): State {
      return State(settings, connectionState, agentStatus, agentIssue, isInlineCompletionLoading)
    }
  }

  var state = State(
    project.service<SettingsState>().settings(),
    ConnectionService.State.INITIALIZING,
    Status.NOT_INITIALIZED,
    null,
    false
  )
    private set

  init {
    messageBusConnection.subscribe(SettingsState.Listener.TOPIC, object : SettingsState.Listener {
      override fun settingsChanged(settings: SettingsState.Settings) {
        state = state.withSettings(settings)
        publisher.stateChanged(state)
      }
    })
    messageBusConnection.subscribe(ConnectionService.Listener.TOPIC, object : ConnectionService.Listener {
      override fun connectionStateChanged(state: ConnectionService.State) {
        this@CombinedState.state = this@CombinedState.state.withConnectionState(state)
        publisher.stateChanged(this@CombinedState.state)
      }
    })
    messageBusConnection.subscribe(LanguageClient.AgentListener.TOPIC, object : LanguageClient.AgentListener {
      override fun agentStatusChanged(status: Status) {
        state = state.withAgentStatus(status)
        publisher.stateChanged(state)
      }

      override fun agentIssueUpdated(issueList: IssueList) {
        state = issueList.issues.firstOrNull()?.let {
          state.withAgentIssue(it)
        } ?: state.withoutAgentIssue()
        publisher.stateChanged(state)
      }
    })

    messageBusConnection.subscribe(InlineCompletionService.Listener.TOPIC, object : InlineCompletionService.Listener {
      override fun loadingStateChanged(loading: Boolean) {
        state = state.withInlineCompletionLoading(loading)
        publisher.stateChanged(state)
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