package com.tabbyml.intellijtabby.lsp

import com.intellij.ide.plugins.PluginManagerCore
import com.intellij.openapi.Disposable
import com.intellij.openapi.application.ApplicationInfo
import com.intellij.openapi.components.serviceOrNull
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.extensions.PluginId
import com.intellij.openapi.project.Project
import com.intellij.openapi.project.guessProjectDir
import com.intellij.openapi.vfs.VirtualFileManager
import com.intellij.psi.PsiFile
import com.intellij.psi.PsiManager
import com.intellij.psi.codeStyle.CodeStyleSettingsManager
import com.intellij.util.messages.Topic
import com.tabbyml.intellijtabby.git.GitProvider
import com.tabbyml.intellijtabby.lsp.protocol.*
import com.tabbyml.intellijtabby.lsp.protocol.ClientCapabilities
import com.tabbyml.intellijtabby.lsp.protocol.ClientInfo
import com.tabbyml.intellijtabby.lsp.protocol.InitializeParams
import com.tabbyml.intellijtabby.lsp.protocol.InitializeResult
import com.tabbyml.intellijtabby.lsp.protocol.ServerInfo
import com.tabbyml.intellijtabby.lsp.protocol.TextDocumentClientCapabilities
import com.tabbyml.intellijtabby.lsp.protocol.server.LanguageServer
import com.tabbyml.intellijtabby.safeSyncPublisher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.future.await
import kotlinx.coroutines.launch
import org.eclipse.lsp4j.*
import java.util.concurrent.CompletableFuture

class LanguageClient(private val project: Project) : com.tabbyml.intellijtabby.lsp.protocol.client.LanguageClient(),
  Disposable {
  private val logger = Logger.getInstance(LanguageClient::class.java)
  private val scope = CoroutineScope(Dispatchers.IO)
  private val virtualFileManager = VirtualFileManager.getInstance()
  private val psiManager = PsiManager.getInstance(project)
  private val gitProvider = project.serviceOrNull<GitProvider>()
  private val configurationSync = ConfigurationSync(project)
  private val textDocumentSync = TextDocumentSync(project)

  override fun buildInitializeParams(): InitializeParams {
    val appInfo = ApplicationInfo.getInstance()
    val appVersion = appInfo.fullVersion
    val appName = appInfo.fullApplicationName.replace(appVersion, "").trim()
    val pluginId = "com.tabbyml.intellij-tabby"
    val pluginVersion = PluginManagerCore.getPlugin(PluginId.getId(pluginId))?.version
    val params = InitializeParams(
      processId = ProcessHandle.current().pid().toInt(), clientInfo = ClientInfo(
        name = appName,
        version = appVersion,
        tabbyPlugin = ClientInfo.TabbyPluginInfo(
          name = pluginId,
          version = pluginVersion,
        ),
      ), initializationOptions = InitializationOptions(
        config = configurationSync.getConfiguration()
      ), capabilities = ClientCapabilities(
        textDocument = TextDocumentClientCapabilities(
          synchronization = SynchronizationCapabilities(),
          inlineCompletion = InlineCompletionCapabilities(),
        ),
        workspace = WorkspaceClientCapabilities().apply {
          workspaceFolders = true
          configuration = true
          didChangeConfiguration = DidChangeConfigurationCapabilities()
        },
        tabby = TabbyClientCapabilities(
          agent = true,
          gitProvider = gitProvider?.isSupported(),
          editorOptions = true,
        ),
      ), workspaceFolders = getWorkspaceFolders()
    )
    logger.info("Initialize params: $params")
    return params
  }

  override fun processInitializeResult(server: LanguageServer, result: InitializeResult?) {
    configurationSync.startSync(server)
    textDocumentSync.startSync(server)
    scope.launch {
      project.safeSyncPublisher(AgentListener.TOPIC)?.agentStatusChanged(server.agentFeature.status().await())
      project.safeSyncPublisher(AgentListener.TOPIC)?.agentIssueUpdated(server.agentFeature.issues().await())
      project.safeSyncPublisher(AgentListener.TOPIC)?.agentServerInfoUpdated(server.agentFeature.serverInfo().await())
    }
  }

  override fun didChangeStatus(params: DidChangeStatusParams) {
    project.safeSyncPublisher(AgentListener.TOPIC)?.agentStatusChanged(params.status)
  }

  override fun didUpdateIssues(params: DidUpdateIssueParams) {
    project.safeSyncPublisher(AgentListener.TOPIC)?.agentIssueUpdated(params)
  }

  override fun didUpdateServerInfo(params: DidUpdateServerInfoParams) {
    project.safeSyncPublisher(AgentListener.TOPIC)?.agentServerInfoUpdated(params.serverInfo)
  }

  override fun editorOptions(params: EditorOptionsParams): CompletableFuture<EditorOptions?> {
    val codeStyleSettingsManager = CodeStyleSettingsManager.getInstance(project)
    val indentation = findPsiFile(params.uri)?.language?.let {
      codeStyleSettingsManager.mainProjectCodeStyle?.getCommonSettings(it)?.indentOptions
    }?.let {
      if (it.USE_TAB_CHARACTER) {
        "\t"
      } else {
        " ".repeat(it.INDENT_SIZE)
      }
    }
    return CompletableFuture<EditorOptions?>().apply {
      complete(EditorOptions(indentation = indentation))
    }
  }

  override fun gitRepository(params: GitRepositoryParams): CompletableFuture<GitRepository?> {
    val repository = gitProvider?.getRepository(params.uri)?.let { repo ->
      GitRepository(root = repo.root, remotes = repo.remotes?.map {
        GitRepository.Remote(
          name = it.name,
          url = it.url,
        )
      })
    }
    return CompletableFuture<GitRepository?>().apply {
      complete(repository)
    }
  }

  override fun gitDiff(params: GitDiffParams): CompletableFuture<GitDiffResult?> {
    val result = gitProvider?.diff(params.repository, params.cached)?.let {
      GitDiffResult(diff = it)
    }
    return CompletableFuture<GitDiffResult?>().apply {
      complete(result)
    }
  }

  override fun registerCapability(params: RegistrationParams): CompletableFuture<Void> {
    // nothing to do for now
    return CompletableFuture<Void>().apply { complete(null) }
  }

  override fun unregisterCapability(params: UnregistrationParams): CompletableFuture<Void> {
    // nothing to do for now
    return CompletableFuture<Void>().apply { complete(null) }
  }

  override fun configuration(params: Any): CompletableFuture<List<ClientProvidedConfig>?> {
    return CompletableFuture<List<ClientProvidedConfig>?>().apply {
      complete(listOf(configurationSync.getConfiguration()))
    }
  }

  override fun workspaceFolders(): CompletableFuture<List<WorkspaceFolder>> {
    return CompletableFuture<List<WorkspaceFolder>>().apply {
      complete(getWorkspaceFolders())
    }
  }

  override fun logMessage(params: MessageParams) {
    when (params.type) {
      MessageType.Error -> logger.warn(params.message)
      MessageType.Warning -> logger.warn(params.message)
      MessageType.Info -> logger.info(params.message)
      MessageType.Log -> logger.debug(params.message)
      null -> return
    }
  }

  override fun logTrace(params: LogTraceParams) {
    logger.trace("${params.message}\n${params.verbose}")
  }

  override fun dispose() {
    configurationSync.dispose()
    textDocumentSync.dispose()
  }

  private fun findPsiFile(fileUri: String): PsiFile? {
    return virtualFileManager.findFileByUrl(fileUri)?.let { psiManager.findFileWithReadLock(it) }
  }

  private fun getWorkspaceFolders(): List<WorkspaceFolder> {
    return project.guessProjectDir()?.let {
      listOf(WorkspaceFolder(it.url, project.name))
    } ?: listOf()
  }

  interface AgentListener {
    fun agentStatusChanged(status: String) {}
    fun agentIssueUpdated(issueList: IssueList) {}
    fun agentServerInfoUpdated(serverInfo: ServerInfo) {}

    companion object {
      @Topic.ProjectLevel
      val TOPIC = Topic(AgentListener::class.java, Topic.BroadcastDirection.NONE)
    }
  }
}