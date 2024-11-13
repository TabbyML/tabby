package com.tabbyml.intellijtabby.lsp

import com.intellij.ide.plugins.PluginManagerCore
import com.intellij.openapi.Disposable
import com.intellij.openapi.application.ApplicationInfo
import com.intellij.openapi.components.serviceOrNull
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.editor.Document
import com.intellij.openapi.extensions.PluginId
import com.intellij.openapi.project.Project
import com.intellij.openapi.project.guessProjectDir
import com.intellij.openapi.util.TextRange
import com.intellij.psi.codeStyle.CodeStyleSettingsManager
import com.intellij.util.messages.Topic
import com.tabbyml.intellijtabby.findDocument
import com.tabbyml.intellijtabby.findPsiFile
import com.tabbyml.intellijtabby.findVirtualFile
import com.tabbyml.intellijtabby.git.GitProvider
import com.tabbyml.intellijtabby.languageSupport.LanguageSupportProvider
import com.tabbyml.intellijtabby.languageSupport.LanguageSupportService
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
  private val gitProvider = project.serviceOrNull<GitProvider>()
  private val languageSupportService = project.serviceOrNull<LanguageSupportService>()
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
          workspaceFileSystem = true,
          languageSupport = languageSupportService != null,
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
    val indentation = project.findPsiFile(params.uri)?.language?.let {
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

  override fun readFile(params: ReadFileParams): CompletableFuture<ReadFileResult?> {
    val file = project.findVirtualFile(params.uri) ?: return CompletableFuture.completedFuture(null)
    when (params.format) {
      ReadFileParams.Format.TEXT -> {
        val document = project.findDocument(file) ?: return CompletableFuture.completedFuture(null)
        val text = if (params.range != null) {
          document.getText(
            TextRange(
              offsetInDocument(document, params.range.start),
              offsetInDocument(document, params.range.end)
            )
          )
        } else {
          document.text
        }
        return CompletableFuture.completedFuture(ReadFileResult(text))
      }

      else -> {
        return CompletableFuture.completedFuture(null)
      }
    }
  }

  override fun declaration(params: DeclarationParams): CompletableFuture<List<LocationLink>?> {
    return CompletableFuture<List<LocationLink>?>().completeAsync {
      val virtualFile = project.findVirtualFile(params.textDocument.uri) ?: return@completeAsync null
      val document = project.findDocument(virtualFile) ?: return@completeAsync null
      val psiFile = project.findPsiFile(virtualFile) ?: return@completeAsync null
      val languageSupport = languageSupportService ?: return@completeAsync null
      languageSupport.provideDeclaration(
        LanguageSupportProvider.FilePosition(psiFile, offsetInDocument(document, params.position))
      )?.mapNotNull {
        val targetUri = it.file.virtualFile.url
        val targetDocument = project.findDocument(it.file.virtualFile) ?: return@mapNotNull null
        val range = Range(
          positionInDocument(targetDocument, it.range.startOffset),
          positionInDocument(targetDocument, it.range.endOffset)
        )
        LocationLink(targetUri, range, range)
      }
    }
  }

  override fun semanticTokensRange(params: SemanticTokensRangeParams): CompletableFuture<SemanticTokensRangeResult?> {
    return CompletableFuture<SemanticTokensRangeResult?>().completeAsync {
      val virtualFile = project.findVirtualFile(params.textDocument.uri) ?: return@completeAsync null
      val document = project.findDocument(virtualFile) ?: return@completeAsync null
      val psiFile = project.findPsiFile(virtualFile) ?: return@completeAsync null
      val languageSupport = languageSupportService ?: return@completeAsync null
      languageSupport.provideSemanticTokensRange(
        LanguageSupportProvider.FileRange(
          psiFile,
          TextRange(
            offsetInDocument(document, params.range.start),
            offsetInDocument(document, params.range.end)
          )
        )
      )?.let {
        encodeSemanticTokens(document, it)
      }
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

  private fun getWorkspaceFolders(): List<WorkspaceFolder> {
    return project.guessProjectDir()?.let {
      listOf(WorkspaceFolder(it.url, project.name))
    } ?: listOf()
  }

  private fun encodeSemanticTokens(
    document: Document,
    tokens: List<LanguageSupportProvider.SemanticToken>
  ): SemanticTokensRangeResult {
    val tokenTypesLegend = mutableListOf<String>()
    val tokenModifiersLegend = mutableListOf<String>()
    val data = mutableListOf<Int>()
    var line = 0
    var character = 0
    for (token in tokens.sortedBy { it.range.startOffset }) {
      val position = positionInDocument(document, token.range.startOffset)
      val deltaLine = position.line - line
      line = position.line
      if (deltaLine != 0) {
        character = 0
      }
      val deltaCharacter = position.character - character
      character = position.character
      val length = token.range.endOffset - token.range.startOffset
      val tokenType = tokenTypesLegend.indexOf(token.type).let {
        if (it == -1) {
          tokenTypesLegend.add(token.type)
          tokenTypesLegend.size - 1
        } else {
          it
        }
      }
      val tokenModifiers = token.modifiers.map { modifier ->
        tokenModifiersLegend.indexOf(modifier).let {
          if (it == -1) {
            tokenModifiersLegend.add(modifier)
            tokenModifiersLegend.size - 1
          } else {
            it
          }
        }
      }.fold(0) { acc, i ->
        acc or (1 shl i)
      }

      data.add(deltaLine)
      data.add(deltaCharacter)
      data.add(length)
      data.add(tokenType)
      data.add(tokenModifiers)
    }
    return SemanticTokensRangeResult(
      legend = SemanticTokensRangeResult.SemanticTokensLegend(tokenTypesLegend, tokenModifiersLegend),
      tokens = SemanticTokensRangeResult.SemanticTokens(data = data)
    )
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