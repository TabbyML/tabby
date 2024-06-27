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
import com.tabbyml.intellijtabby.lsp.protocol.Status
import com.tabbyml.intellijtabby.notifications.notifyAuthRequired
import com.tabbyml.intellijtabby.settings.SettingsService

@Service(Service.Level.PROJECT)
class CombinedState(private val project: Project) : Disposable {

  private val messageBusConnection = project.messageBus.connect()
  private val publisher = project.messageBus.syncPublisher(Listener.TOPIC)

  data class State(
    val settings: SettingsService.Settings,
    val connectionState: ConnectionService.State,
    val agentStatus: String,
    val agentIssue: String?,
    val isInlineCompletionLoading: Boolean,
  ) {
    fun withSettings(settings: SettingsService.Settings): State {
      return State(settings, connectionState, agentStatus, agentIssue, isInlineCompletionLoading)
    }

    fun withConnectionState(connectionState: ConnectionService.State): State {
      return State(settings, connectionState, agentStatus, agentIssue, isInlineCompletionLoading)
    }

    fun withAgentStatus(agentStatus: String): State {
      return State(settings, connectionState, agentStatus, agentIssue, isInlineCompletionLoading)
    }

    fun withAgentIssue(currentIssue: String?): State {
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
    service<SettingsService>().settings(),
    ConnectionService.State.INITIALIZING,
    Status.NOT_INITIALIZED,
    null,
    false
  )
    private set

  init {
    messageBusConnection.subscribe(SettingsService.Listener.TOPIC, object : SettingsService.Listener {
      override fun settingsChanged(settings: SettingsService.Settings) {
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
      override fun agentStatusChanged(status: String) {
        state = state.withAgentStatus(status)
        publisher.stateChanged(state)

        if (status == Status.UNAUTHORIZED) {
          notifyAuthRequired()
        }
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