package com.tabbyml.intellijtabby.agent

import com.intellij.ide.BrowserUtil
import com.intellij.ide.plugins.PluginManagerCore
import com.intellij.lang.Language
import com.intellij.notification.Notification
import com.intellij.notification.NotificationType
import com.intellij.notification.Notifications
import com.intellij.openapi.Disposable
import com.intellij.openapi.actionSystem.ActionManager
import com.intellij.openapi.application.ApplicationInfo
import com.intellij.openapi.application.ReadAction
import com.intellij.openapi.application.invokeLater
import com.intellij.openapi.components.Service
import com.intellij.openapi.components.service
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.extensions.PluginId
import com.intellij.openapi.progress.ProgressIndicator
import com.intellij.psi.PsiDocumentManager
import com.intellij.psi.PsiFile
import com.tabbyml.intellijtabby.settings.ApplicationSettingsState
import io.ktor.util.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

@Service
class AgentService : Disposable {
  private val logger = Logger.getInstance(AgentService::class.java)
  private var agent: Agent = Agent()
  val scope: CoroutineScope = CoroutineScope(Dispatchers.IO)

  private var initFailedNotification: Notification? = null
  var authNotification: Notification? = null
    private set
  var issueNotification: Notification? = null
    private set

  private var completionResponseWarningShown = false

  enum class Status {
    INITIALIZING,
    INITIALIZATION_FAILED,
  }

  private var initResultFlow: MutableStateFlow<Boolean?> = MutableStateFlow(null)
  val status = initResultFlow.combine(agent.status) { initResult, agentStatus ->
    if (initResult == null) {
      Status.INITIALIZING
    } else if (initResult) {
      agentStatus
    } else {
      Status.INITIALIZATION_FAILED
    }
  }.stateIn(scope, SharingStarted.Eagerly, Status.INITIALIZING)

  val currentIssue get() = agent.currentIssue

  init {
    val settings = service<ApplicationSettingsState>()
    scope.launch {
      val config = createAgentConfig(settings.data)
      val clientProperties = createClientProperties(settings.data)
      try {
        agent.open()
        agent.initialize(config, clientProperties)
        initResultFlow.value = true
        logger.info("Agent init done.")
      } catch (e: Exception) {
        initResultFlow.value = false
        logger.warn("Agent init failed: $e")

        val notification = Notification(
          "com.tabbyml.intellijtabby.notification.warning",
          "Tabby initialization failed",
          "${e.message}",
          NotificationType.ERROR,
        )
        notification.addAction(ActionManager.getInstance().getAction("Tabby.OpenOnlineDocs"))
        invokeLater {
          initFailedNotification?.expire()
          initFailedNotification = notification
          Notifications.Bus.notify(notification)
        }
      }
    }

    scope.launch {
      settings.serverEndpointState.collect {
        setEndpoint(it)
      }
    }

    scope.launch {
      settings.completionTriggerModeState.collect {
        updateClientProperties("user", "intellij.triggerMode", it)
      }
    }

    scope.launch {
      settings.isAnonymousUsageTrackingDisabledState.collect {
        updateConfig("anonymousUsageTracking.disable", it)
      }
    }

    scope.launch {
      agent.authRequiredEvent.collect {
        logger.info("Will show auth required notification.")
        val notification = Notification(
          "com.tabbyml.intellijtabby.notification.warning",
          "Authorization required for Tabby server",
          NotificationType.WARNING,
        )
        notification.addAction(ActionManager.getInstance().getAction("Tabby.OpenAuthPage"))
        invokeLater {
          authNotification?.expire()
          authNotification = notification
          Notifications.Bus.notify(notification)
        }
      }
    }


    scope.launch {
      agent.status.collect { status ->
        if (status == Agent.Status.READY) {
          completionResponseWarningShown = false
        }
      }
    }

    scope.launch {
      agent.currentIssue.collect { issueName ->
        val message = when (issueName) {
          "connectionFailed" -> "Cannot connect to Tabby server"
          "slowCompletionResponseTime" -> if (!completionResponseWarningShown) {
            completionResponseWarningShown = true
            "Completion requests appear to take too much time"
          } else {
            return@collect
          }
          "highCompletionTimeoutRate" -> if (!completionResponseWarningShown) {
            completionResponseWarningShown = true
            "Most completion requests timed out"
          } else {
            return@collect
          }
          else -> {
            invokeLater {
              issueNotification?.expire()
            }
            return@collect
          }
        }
        val notification = Notification(
          "com.tabbyml.intellijtabby.notification.warning",
          message,
          NotificationType.WARNING,
        )
        notification.addAction(ActionManager.getInstance().getAction("Tabby.CheckIssueDetail"))
        invokeLater {
          issueNotification?.expire()
          issueNotification = notification
          Notifications.Bus.notify(notification)
        }
      }
    }
  }

  private fun createAgentConfig(state: ApplicationSettingsState.State): Agent.Config {
    return Agent.Config(
      server = if (state.serverEndpoint.isNotBlank()) {
        Agent.Config.Server(
          endpoint = state.serverEndpoint,
        )
      } else {
        null
      },
      anonymousUsageTracking = if (state.isAnonymousUsageTrackingDisabled) {
        Agent.Config.AnonymousUsageTracking(
          disabled = true,
        )
      } else {
        null
      },
    )
  }

  private fun createClientProperties(state: ApplicationSettingsState.State): Agent.ClientProperties {
    val appInfo = ApplicationInfo.getInstance()
    val appVersion = appInfo.fullVersion
    val appName = appInfo.fullApplicationName.replace(appVersion, "").trim()
    val pluginId = "com.tabbyml.intellij-tabby"
    val pluginVersion = PluginManagerCore.getPlugin(PluginId.getId(pluginId))?.version
    val client = "$appName $pluginId $pluginVersion"
    return Agent.ClientProperties(
      user = mapOf(
        "intellij" to mapOf(
          "triggerMode" to state.completionTriggerMode,
        ),
      ),
      session = mapOf(
        "client" to client,
        "ide" to mapOf("name" to appName, "version" to appVersion),
        "tabby_plugin" to mapOf("name" to pluginId, "version" to pluginVersion),
      ),
    )
  }

  private suspend fun waitForInitialized() {
    agent.status.first { it != Agent.Status.NOT_INITIALIZED }
  }

  private suspend fun updateClientProperties(type: String, key: String, config: Any) {
    waitForInitialized()
    agent.updateClientProperties(type, key, config)
  }

  private suspend fun updateConfig(key: String, config: Any) {
    waitForInitialized()
    agent.updateConfig(key, config)
  }

  private suspend fun clearConfig(key: String) {
    waitForInitialized()
    agent.clearConfig(key)
  }

  suspend fun setEndpoint(endpoint: String) {
    if (endpoint.isNotBlank()) {
      updateConfig("server.endpoint", endpoint)
    } else {
      clearConfig("server.endpoint")
    }
  }

  suspend fun getConfig(): Agent.Config {
    waitForInitialized()
    return agent.getConfig()
  }

  suspend fun provideCompletion(editor: Editor, offset: Int, manually: Boolean = false): Agent.CompletionResponse? {
    waitForInitialized()
    return ReadAction.compute<PsiFile, Throwable> {
      editor.project?.let { project ->
        PsiDocumentManager.getInstance(project).getPsiFile(editor.document)
      }
    }?.let { file ->
      agent.provideCompletions(
        Agent.CompletionRequest(
          file.virtualFile.path,
          file.getLanguageId(),
          editor.document.text,
          offset,
          manually,
        )
      )
    }
  }

  suspend fun postEvent(event: Agent.LogEventRequest) {
    waitForInitialized()
    agent.postEvent(event)
  }

  suspend fun requestAuth(progress: ProgressIndicator) {
    waitForInitialized()
    progress.isIndeterminate = true
    progress.text = "Generating authorization url..."
    val authUrlResponse = agent.requestAuthUrl()
    val notification = if (authUrlResponse != null) {
      BrowserUtil.browse(authUrlResponse.authUrl)
      progress.text = "Waiting for authorization from browser..."
      agent.waitForAuthToken(authUrlResponse.code)
      if (agent.status.value == Agent.Status.READY) {
        Notification(
          "com.tabbyml.intellijtabby.notification.info",
          "Congrats, you're authorized, start to use Tabby now.",
          NotificationType.INFORMATION
        )
      } else {
        Notification(
          "com.tabbyml.intellijtabby.notification.warning",
          "Connection error, please check settings and try again.",
          NotificationType.WARNING
        )
      }
    } else {
      Notification(
        "com.tabbyml.intellijtabby.notification.info", "You are already authorized.", NotificationType.INFORMATION
      )
    }
    invokeLater {
      authNotification?.expire()
      authNotification = notification
      Notifications.Bus.notify(notification)
    }
  }

  suspend fun getCurrentIssueDetail(): Map<String, Any>? {
    waitForInitialized()
    return agent.getIssueDetail(Agent.GetIssueDetailOptions(name = currentIssue.value))
  }

  suspend fun getServerHealthState(): Map<String, Any>? {
    waitForInitialized()
    return agent.getServerHealthState()
  }

  override fun dispose() {
    runBlocking {
      runCatching {
        agent.finalize()
      }
    }
    agent.close()
  }

  companion object {
    // Language id: https://code.visualstudio.com/docs/languages/identifiers
    private fun PsiFile.getLanguageId(): String {
      if (this.language != Language.ANY && this.language.id.toLowerCasePreservingASCIIRules() !in arrayOf(
          "txt",
          "text",
          "textmate"
        )
      ) {
        if (languageIdMap.containsKey(this.language.id)) {
          return languageIdMap[this.language.id]!!
        }
        return this.language.id.toLowerCasePreservingASCIIRules().replace("#", "sharp").replace("++", "pp")
          .replace(" ", "")
      }
      return if (filetypeMap.containsKey(this.fileType.defaultExtension)) {
        filetypeMap[this.fileType.defaultExtension]!!
      } else {
        this.fileType.defaultExtension.toLowerCasePreservingASCIIRules()
      }
    }

    private val languageIdMap = mapOf(
      "ObjectiveC" to "objective-c",
      "ObjectiveC++" to "objective-cpp",
    )
    private val filetypeMap = mapOf(
      "py" to "python",
      "js" to "javascript",
      "cjs" to "javascript",
      "mjs" to "javascript",
      "jsx" to "javascriptreact",
      "ts" to "typescript",
      "tsx" to "typescriptreact",
      "kt" to "kotlin",
      "md" to "markdown",
      "cc" to "cpp",
      "cs" to "csharp",
      "m" to "objective-c",
      "mm" to "objective-cpp",
      "sh" to "shellscript",
      "zsh" to "shellscript",
      "bash" to "shellscript",
      "txt" to "plaintext",
    )
  }
}