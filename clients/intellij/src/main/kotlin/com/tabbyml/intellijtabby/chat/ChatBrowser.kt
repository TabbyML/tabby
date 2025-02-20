package com.tabbyml.intellijtabby.chat

import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.intellij.ide.BrowserUtil
import com.intellij.ide.plugins.PluginManagerCore
import com.intellij.openapi.application.invokeLater
import com.intellij.openapi.application.runReadAction
import com.intellij.openapi.command.WriteCommandAction
import com.intellij.openapi.components.service
import com.intellij.openapi.components.serviceOrNull
import com.intellij.openapi.diagnostic.logger
import com.intellij.openapi.editor.Document
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.colors.EditorColors
import com.intellij.openapi.editor.colors.EditorColorsListener
import com.intellij.openapi.editor.colors.EditorColorsManager
import com.intellij.openapi.editor.colors.EditorFontType
import com.intellij.openapi.editor.event.SelectionEvent
import com.intellij.openapi.extensions.PluginId
import com.intellij.openapi.fileEditor.FileEditorManager
import com.intellij.openapi.fileEditor.FileEditorManagerEvent
import com.intellij.openapi.fileEditor.FileEditorManagerListener
import com.intellij.openapi.fileEditor.OpenFileDescriptor
import com.intellij.openapi.progress.util.BackgroundTaskUtil
import com.intellij.openapi.project.Project
import com.intellij.openapi.project.guessProjectDir
import com.intellij.openapi.util.SystemInfo
import com.intellij.openapi.vfs.VirtualFile
import com.intellij.ui.JBColor
import com.intellij.ui.jcef.JBCefBrowser
import com.intellij.ui.jcef.JBCefBrowserBase
import com.intellij.ui.jcef.JBCefJSQuery
import com.tabbyml.intellijtabby.events.CombinedState
import com.tabbyml.intellijtabby.events.SelectionListener
import com.tabbyml.intellijtabby.findVirtualFile
import com.tabbyml.intellijtabby.git.GitProvider
import com.tabbyml.intellijtabby.lsp.ConnectionService
import com.tabbyml.intellijtabby.lsp.ConnectionService.InitializationException
import com.tabbyml.intellijtabby.lsp.positionInDocument
import com.tabbyml.intellijtabby.lsp.protocol.Config
import com.tabbyml.intellijtabby.lsp.protocol.StatusInfo
import com.tabbyml.intellijtabby.lsp.protocol.StatusRequestParams
import io.github.z4kn4fein.semver.Version
import io.github.z4kn4fein.semver.constraints.Constraint
import io.github.z4kn4fein.semver.constraints.satisfiedBy
import io.ktor.http.*
import kotlinx.coroutines.*
import org.cef.browser.CefBrowser
import org.cef.handler.CefLoadHandlerAdapter
import java.awt.Color
import java.awt.Toolkit
import java.awt.datatransfer.StringSelection
import java.io.File
import java.util.*
import java.util.concurrent.CompletableFuture


class ChatBrowser(private val project: Project) : JBCefBrowser(
  createBuilder()
    .setOffScreenRendering(
      when {
        SystemInfo.isWindows -> false
        SystemInfo.isMac -> false
        SystemInfo.isLinux -> true
        else -> false
      }
    )
    .setEnableOpenDevToolsMenuItem(true)
) {
  private val logger = logger<ChatBrowser>()
  private val gson = Gson()
  private val combinedState = project.service<CombinedState>()
  private val gitProvider = project.service<GitProvider>()
  private val fileEditorManager = FileEditorManager.getInstance(project)
  private val messageBusConnection = project.messageBus.connect()

  private val scope = CoroutineScope(Dispatchers.IO)
  private var syncChatPanelActiveSelectionJob: Job? = null

  private suspend fun getServer() = project.serviceOrNull<ConnectionService>()?.getServerAsync()

  private var currentConfig: Config.ServerConfig? = null
  private var isChatPanelLoaded = false
  private val pendingScripts: MutableList<String> = mutableListOf()

  init {
    component.isVisible = false
    val bgColor = calcComponentBgColor()
    component.background = bgColor
    setPageBackgroundColor("hsl(${bgColor.toHsl()})")

    val tabbyThreadsScript = loadTabbyThreadsScript()
    val htmlContent = loadHtmlContent(tabbyThreadsScript)
    loadHTML(htmlContent)

    jbCefClient.addLoadHandler(object : CefLoadHandlerAdapter() {
      override fun onLoadingStateChange(
        browser: CefBrowser?,
        isLoading: Boolean,
        canGoBack: Boolean,
        canGoForward: Boolean
      ) {
        if (browser != null && !isLoading) {
          handleLoaded()
        }
      }
    }, cefBrowser)

    messageBusConnection.subscribe(CombinedState.Listener.TOPIC, object : CombinedState.Listener {
      override fun stateChanged(state: CombinedState.State) {
        reloadContent()
      }
    })

    messageBusConnection.subscribe(FileEditorManagerListener.FILE_EDITOR_MANAGER, object : FileEditorManagerListener {
      override fun selectionChanged(event: FileEditorManagerEvent) {
        syncChatPanelActiveSelectionJob?.cancel()
        syncChatPanelActiveSelectionJob = scope.launch {
          BackgroundTaskUtil.executeOnPooledThread(this@ChatBrowser) {
            val context = getActiveEditorFileContext()
            chatPanelUpdateActiveSelection(context)
          }
        }
      }
    })

    messageBusConnection.subscribe(SelectionListener.TOPIC, object : SelectionListener {
      override fun selectionChanged(editor: Editor, event: SelectionEvent) {
        if (editor == fileEditorManager.selectedTextEditor) {
          syncChatPanelActiveSelectionJob?.cancel()
          syncChatPanelActiveSelectionJob = scope.launch {
            delay(100)
            BackgroundTaskUtil.executeOnPooledThread(this@ChatBrowser) {
              val context = getActiveEditorFileContext()
              chatPanelUpdateActiveSelection(context)
            }
          }
        }
      }
    })
    messageBusConnection.subscribe(EditorColorsManager.TOPIC, EditorColorsListener {
      BackgroundTaskUtil.executeOnPooledThread(this) {
        logger.debug("EditorColorsManager globalSchemeChange received, updating style.")
        Thread.sleep(100)
        jsApplyStyle()
        chatPanelUpdateTheme()
      }
    })
  }

  fun explainSelectedText() {
    BackgroundTaskUtil.executeOnPooledThread(this) {
      chatPanelExecuteCommand(ChatCommand.EXPLAIN)
    }
  }

  fun fixSelectedText() {
    BackgroundTaskUtil.executeOnPooledThread(this) {
      // FIXME(@icycodes): collect the diagnostic message provided by IDE
      chatPanelExecuteCommand(ChatCommand.FIX)
    }
  }

  fun generateDocsForSelectedText() {
    BackgroundTaskUtil.executeOnPooledThread(this) {
      chatPanelExecuteCommand(ChatCommand.GENERATE_DOCS)
    }
  }

  fun generateTestsForSelectedText() {
    BackgroundTaskUtil.executeOnPooledThread(this) {
      chatPanelExecuteCommand(ChatCommand.GENERATE_TESTS)
    }
  }

  // default: use selection if available, otherwise use the whole file
  // selection: use selection if available, otherwise return null
  // file: use the whole file
  enum class RangeStrategy {
    DEFAULT, SELECTION, FILE
  }

  fun addActiveEditorAsContext(rangeStrategy: RangeStrategy = RangeStrategy.DEFAULT) {
    BackgroundTaskUtil.executeOnPooledThread(this) {
      val context = getActiveEditorFileContext(rangeStrategy) ?: return@executeOnPooledThread
      chatPanelAddRelevantContext(context)
    }
  }

  private fun virtualFileToFilepath(virtualFile: VirtualFile): Filepath {
    val uri = virtualFile.url

    val workspaceDir = project.guessProjectDir()?.url
    val gitRepo = gitProvider.getRepository(uri)
    val gitUrl = gitRepo?.let { getDefaultRemoteUrl(it) }

    return if (gitUrl != null && uri.startsWith(gitRepo.root)) {
      val relativePath = uri.substringAfter(gitRepo.root).trimStart(File.separatorChar)
      FilepathInGitRepository(
        filepath = relativePath,
        gitUrl = gitUrl,
      )
    } else if (workspaceDir != null && uri.startsWith(workspaceDir)) {
      FilepathInWorkspace(
        filepath = uri.substringAfter(workspaceDir).trimStart(File.separatorChar),
        baseDir = workspaceDir,
      )
    } else {
      FilepathUri(uri = uri)
    }
  }

  private fun findVirtualFile(filepath: Filepath): VirtualFile? {
    return when (filepath.kind) {
      Filepath.Kind.URI -> {
        val filepathUri = filepath as FilepathUri
        project.findVirtualFile(filepathUri.uri) ?: project.guessProjectDir()?.url?.let {
          project.findVirtualFile(it.appendUrlPathSegments(filepathUri.uri))
        }
      }

      Filepath.Kind.WORKSPACE -> {
        val filepathInWorkspace = filepath as FilepathInWorkspace
        filepathInWorkspace.baseDir.let {
          project.findVirtualFile(it.appendUrlPathSegments(filepathInWorkspace.filepath))
        } ?: project.guessProjectDir()?.url?.let {
          project.findVirtualFile(it.appendUrlPathSegments(filepathInWorkspace.filepath))
        }
      }

      Filepath.Kind.GIT -> {
        val filepathInGit = filepath as FilepathInGitRepository
        gitRemoteUrlToLocalRoot[filepathInGit.gitUrl]?.let {
          project.findVirtualFile(it.appendUrlPathSegments(filepathInGit.filepath))
        } ?: project.guessProjectDir()?.url?.let {
          project.findVirtualFile(it.appendUrlPathSegments(filepathInGit.filepath))
        }
      }

      else -> {
        null
      }
    }
  }

  private fun getActiveEditorFileContext(rangeStrategy: RangeStrategy = RangeStrategy.DEFAULT): EditorFileContext? {
    val editor = fileEditorManager.selectedTextEditor ?: return null
    val virtualFile = editor.virtualFile ?: return null

    val context = runReadAction {
      val document = editor.document
      val selectionModel = editor.selectionModel
      val useSelectedText = rangeStrategy == RangeStrategy.SELECTION
          || (rangeStrategy == RangeStrategy.DEFAULT && selectionModel.hasSelection())
      if (useSelectedText) {
        val text = selectionModel.selectedText.takeUnless { it.isNullOrBlank() } ?: return@runReadAction null
        Pair(
          text,
          PositionRange(
            positionOneBasedInDocument(document, selectionModel.selectionStart),
            positionOneBasedInDocument(document, selectionModel.selectionEnd),
          )
        )
      } else {
        val text = document.text.takeUnless { it.isBlank() } ?: return@runReadAction null
        Pair(
          text,
          null,
        )
      }
    } ?: return null

    val filepath = virtualFileToFilepath(virtualFile)
    val editorFileContext = EditorFileContext(
      filepath = filepath,
      range = context.second,
      content = context.first,
    )

    logger.debug("Collected active editor file context: $editorFileContext")
    return editorFileContext
  }

  private fun openInEditor(fileLocation: FileLocation): Boolean {
    val filepath = fileLocation.filepath
    val virtualFile = findVirtualFile(filepath) ?: return false

    val position = when (val location = fileLocation.location) {
      is Number -> {
        Position(location.toInt() - 1, 0)
      }

      is Position -> {
        Position(location.line - 1, location.character - 1)
      }

      is LineRange -> {
        Position(location.start - 1, 0)
      }

      is PositionRange -> {
        Position(location.start.line - 1, location.start.character - 1)
      }

      else -> {
        null
      }
    }

    invokeLater {
      val descriptor = OpenFileDescriptor(
        project,
        virtualFile,
        position?.line?.coerceAtLeast(0) ?: -1,
        position?.character?.coerceAtLeast(0) ?: -1,
      )
      fileEditorManager.openTextEditor(descriptor, true)
    }
    return true
  }

  private fun handleLoaded() {
    jsInjectFunctions()
    jsCreateChatPanelClient()
    jsApplyStyle()
    reloadContent()
    component.isVisible = true
  }

  private val isDarkTheme get() = EditorColorsManager.getInstance().isDarkEditor

  private fun calcComponentBgColor(): Color {
    val editorColorsScheme = EditorColorsManager.getInstance().schemeForCurrentUITheme
    return editorColorsScheme.getColor(EditorColors.CARET_ROW_COLOR)
      ?: if (isDarkTheme) editorColorsScheme.defaultBackground.brighter() else editorColorsScheme.defaultBackground.darker()
  }

  private fun buildCss(): String {
    val editorColorsScheme = EditorColorsManager.getInstance().schemeForCurrentUITheme
    val bgColor = editorColorsScheme.defaultBackground
    val bgActiveColor = calcComponentBgColor()
    val fgColor = editorColorsScheme.defaultForeground
    val borderColor = if (isDarkTheme) JBColor.DARK_GRAY else JBColor.LIGHT_GRAY
    val inputBorderColor = borderColor

    val primaryColor = editorColorsScheme.getAttributes(EditorColors.REFERENCE_HYPERLINK_COLOR).foregroundColor
      ?: if (isDarkTheme) Color(55, 148, 255) else Color(26, 133, 255)
    val primaryFgColor = JBColor.WHITE
    val popoverColor = bgActiveColor
    val popoverFgColor = fgColor
    val accentColor = if (isDarkTheme) Color(4, 57, 94) else bgActiveColor.interpolate(bgActiveColor.darker(), 0.2)
    val accentFgColor = fgColor

    val font = editorColorsScheme.getFont(EditorFontType.PLAIN).fontName
    val fontSize = editorColorsScheme.editorFontSize
    val css = String.format("background-color: hsl(%s);", bgActiveColor.toHsl()) +
        String.format("--background: %s;", bgColor.toHsl()) +
        String.format("--foreground: %s;", fgColor.toHsl()) +
        String.format("--border: %s;", borderColor.toHsl()) +
        String.format("--input: %s;", inputBorderColor.toHsl()) +
        String.format("--primary: %s;", primaryColor.toHsl()) +
        String.format("--primary-foreground: %s;", primaryFgColor.toHsl()) +
        String.format("--popover: %s;", popoverColor.toHsl()) +
        String.format("--popover-foreground: %s;", popoverFgColor.toHsl()) +
        String.format("--accent: %s;", accentColor.toHsl()) +
        String.format("--accent-foreground: %s;", accentFgColor.toHsl()) +
        String.format("font: %s;", font) +
        String.format("font-size: %spx;", fontSize)
    logger.debug("CSS: $css")
    return css
  }

  private fun reloadContent(force: Boolean = false) {
    if (force) {
      scope.launch {
        val server = getServer() ?: return@launch
        server.statusFeature.getStatus(StatusRequestParams(recheckConnection = true)).thenAccept {
          reloadContentInternal(it, true)
        }
      }
    } else {
      reloadContentInternal(combinedState.state.agentStatus)
    }
  }

  private fun reloadContentInternal(statusInfo: StatusInfo?, force: Boolean = false) {
    if (statusInfo == null) {
      currentConfig = null
      showContent("Initializing...")
    } else {
      when (statusInfo.status) {
        StatusInfo.Status.CONNECTING -> {
          currentConfig = null
          showContent("Connecting to Tabby server...")
        }

        StatusInfo.Status.UNAUTHORIZED -> {
          currentConfig = null
          showContent("Authorization required, please set your token in settings.")
        }

        StatusInfo.Status.DISCONNECTED -> {
          currentConfig = null
          showContent("Cannot connect to Tabby server, please check your settings.")
        }

        else -> {
          val health = statusInfo.serverHealth
          val error = checkServerHealth(health)
          if (error != null) {
            currentConfig = null
            showContent(error)
          } else {
            val config = combinedState.state.agentConfig?.server
            if (config == null) {
              currentConfig = null
              showContent("Initializing...")
            } else if (force || currentConfig != config) {
              showContent("Loading chat panel...")
              isChatPanelLoaded = false
              currentConfig = config
              jsLoadChatPanel()
            }
          }
        }
      }
    }
  }

  private fun showContent(message: String? = null) {
    if (message != null) {
      jsShowMessage(message)
      jsShowChatPanel(false)
    } else {
      jsShowMessage(null)
      jsShowChatPanel(true)
    }
  }

  // chat panel api functions

  private fun chatPanelInit() {
    val params =
      listOf(
        mapOf(
          "fetcherOptions" to mapOf(
            "authorization" to currentConfig?.token,
          )
        )
      )
    logger.debug("chatPanelInit: $params")
    jsChatPanelClientInvoke("init", params)
  }

  private fun chatPanelExecuteCommand(command: String) {
    val params = listOf(command)
    logger.debug("chatPanelExecuteCommand: $params")
    jsChatPanelClientInvoke("executeCommand", params)
  }

  private fun chatPanelAddRelevantContext(context: EditorFileContext) {
    val params = listOf(context)

    logger.debug("chatPanelAddRelevantContext: $params")
    jsChatPanelClientInvoke("addRelevantContext", params)
  }

  private fun chatPanelUpdateTheme() {
    val params =
      listOf(
        buildCss(),
        if (isDarkTheme) "dark" else "light",
      )
    logger.debug("chatPanelUpdateTheme: $params")
    jsChatPanelClientInvoke("updateTheme", params)
  }

  private fun chatPanelUpdateActiveSelection(context: EditorFileContext?) {
    val params = listOf(context)
    logger.debug("chatPanelUpdateActiveSelection: $params")
    jsChatPanelClientInvoke("updateActiveSelection", params)
  }

  // js handler functions to inject

  private fun createJsFunction(handler: (List<Any?>) -> Any?): String {
    val jsQuery = JBCefJSQuery.create(this as JBCefBrowserBase)
    jsQuery.addHandler { paramsJson ->
      val params = gson.fromJson(paramsJson, object : TypeToken<List<Any?>>() {})
      val result = handler(params)
      val resultJson = gson.toJson(result)
      return@addHandler JBCefJSQuery.Response(resultJson)
    }
    val injection = jsQuery.inject(
      "paramsJson",
      "function(response) { resolve(JSON.parse(response)); }",
      "function(error_code, error_message) { reject(new Error(error_message)); }",
    )
    return """
      function(...args) {
        return new Promise((resolve, reject) => {
          const paramsJson = JSON.stringify(args);
          $injection
        });
      }
    """.trimIndent().trimStart()
  }

  private val jsReloadContent = createJsFunction { reloadContent(true) }

  private val jsHandleChatPanelRefresh = createJsFunction {
    logger.debug("refresh")
    reloadContent(true)
  }

  private val jsHandleChatPanelOnApplyInEditor = createJsFunction { params ->
    logger.debug("onApplyInEditor: $params")
    val content = params.getOrNull(0) as String? ?: return@createJsFunction Unit
    val editor = fileEditorManager.selectedTextEditor ?: return@createJsFunction Unit
    invokeLater {
      WriteCommandAction.runWriteCommandAction(project) {
        val start = editor.selectionModel.selectionStart
        val end = editor.selectionModel.selectionEnd
        editor.document.replaceString(start, end, content)
        editor.caretModel.moveToOffset(start + content.length)
      }
    }
  }

  private val jsHandleChatPanelOnLoaded = createJsFunction { params ->
    logger.debug("onLoaded: $params")
    val onLoadedParams = params.getOrNull(0) as Map<*, *>?
    val apiVersion = onLoadedParams?.get("apiVersion") as String?
    if (apiVersion != null) {
      val error = checkChatPanelApiVersion(apiVersion)
      if (error != null) {
        showContent(error)
        return@createJsFunction Unit
      }
    }
    isChatPanelLoaded = true
    pendingScripts.forEach { executeJs(it) }
    pendingScripts.clear()
    chatPanelInit()
    chatPanelUpdateTheme()
    showContent()
  }

  private val jsHandleChatPanelOnCopy = createJsFunction { params ->
    logger.debug("onCopy: request: $params")
    val content = params.getOrNull(0) as String? ?: return@createJsFunction Unit
    val stringSelection = StringSelection(content)
    val clipboard = Toolkit.getDefaultToolkit().systemClipboard
    clipboard.setContents(stringSelection, null)
  }

  private val jsHandleChatPanelOnKeyboardEvent = createJsFunction { params ->
    logger.debug("onKeyboardEvent: request: $params")
    // nothing to do
  }

  private val jsHandleChatPanelOpenInEditor = createJsFunction { params ->
    logger.debug("openInEditor: request: $params")
    val fileLocation = params.getOrNull(0)?.asFileLocation() ?: return@createJsFunction false
    return@createJsFunction openInEditor(fileLocation)
  }

  private val jsHandleChatPanelOpenExternal = createJsFunction { params ->
    logger.debug("openExternal: request: $params")
    val url = params.getOrNull(0) as String? ?: return@createJsFunction Unit
    BrowserUtil.browse(url)
  }

  private val jsHandleChatPanelReadWorkspaceGitRepositories = createJsFunction { params ->
    logger.debug("readWorkspaceGitRepositories: request: $params")
    val activeTextEditorUri = fileEditorManager.selectedTextEditor?.virtualFile?.url
    val projectDir = project.guessProjectDir()?.url
    val pathToCheck = activeTextEditorUri ?: projectDir ?: return@createJsFunction null
    val gitRepo = gitProvider.getRepository(pathToCheck)?.let {
      getDefaultRemoteUrl(it)
    }?.let {
      GitRepository(it)
    }
    return@createJsFunction listOfNotNull(gitRepo)
  }

  private val jsHandleChatPanelGetActiveEditorSelection = createJsFunction { params ->
    logger.debug("getActiveEditorSelection: request: $params")
    return@createJsFunction getActiveEditorFileContext()
  }

  // functions to execute js scripts

  private fun executeJs(script: String) {
    cefBrowser.executeJavaScript(script, cefBrowser.url, 0)
  }

  private fun jsInjectFunctions() {
    val script = """
      if (!window.handleReload) {
        window.handleReload = $jsReloadContent
      }
    """.trimIndent().trimStart()
    executeJs(script)
  }

  private fun jsCreateChatPanelClient() {
    val script = """
      if (!window.tabbyChatPanelClient) {
        window.tabbyChatPanelClient = TabbyThreads.createThreadFromIframe(getChatPanel(), {
          expose: {
            refresh: $jsHandleChatPanelRefresh,
            onApplyInEditor: $jsHandleChatPanelOnApplyInEditor,
            onLoaded: $jsHandleChatPanelOnLoaded,
            onCopy: $jsHandleChatPanelOnCopy,
            onKeyboardEvent: $jsHandleChatPanelOnKeyboardEvent,
            openInEditor: $jsHandleChatPanelOpenInEditor,
            openExternal: $jsHandleChatPanelOpenExternal,
            readWorkspaceGitRepositories: $jsHandleChatPanelReadWorkspaceGitRepositories,
            getActiveEditorSelection: $jsHandleChatPanelGetActiveEditorSelection,
          }
        })
      }
    """.trimIndent().trimStart()
    executeJs(script)
  }

  private fun jsApplyStyle() {
    val script = String.format(
      "applyStyle('%s')",
      gson.toJson(mapOf("theme" to if (isDarkTheme) "dark" else "light", "css" to buildCss()))
    )
    executeJs(script)
  }

  private fun jsShowMessage(message: String?) {
    val script = if (message != null) "showMessage('${message}')" else "showMessage()"
    executeJs(script)
  }

  private fun jsShowChatPanel(visible: Boolean) {
    val script = String.format("showChatPanel(%s)", if (visible) "true" else "false")
    executeJs(script)
  }

  private fun jsLoadChatPanel() {
    val config = currentConfig ?: return
    val chatUrl = "${config.endpoint}/chat?client=intellij"
    val script = "loadChatPanel('$chatUrl')"
    executeJs(script)
  }

  private val pendingChatPanelRequest = mutableMapOf<String, CompletableFuture<Any?>>()
  private val jsChatPanelResponseHandlerInjection = JBCefJSQuery.create(this as JBCefBrowserBase).apply {
    addHandler { results ->
      logger.debug("Response from chat panel: $results")
      val parsedResult = gson.fromJson(results, object : TypeToken<List<Any?>>() {})
      val future = pendingChatPanelRequest.remove(parsedResult[0] as String)
      if (parsedResult[1] is String) {
        future?.completeExceptionally(Exception(parsedResult[1] as String))
      } else {
        future?.complete(parsedResult[2])
      }
      return@addHandler JBCefJSQuery.Response("")
    }
  }.inject("results")

  private fun jsChatPanelClientInvoke(method: String, params: List<Any?>): CompletableFuture<Any?> {
    val future = CompletableFuture<Any?>()
    val uuid = UUID.randomUUID().toString()
    pendingChatPanelRequest[uuid] = future
    val paramsJson = escapeCharacters(gson.toJson(params))
    val script = """
      (function() {
        const func = window.tabbyChatPanelClient['$method']
        if (func && typeof func === 'function') {
          const params = JSON.parse('$paramsJson')
          const resultPromise = func(...params)
          if (resultPromise && typeof resultPromise.then === 'function') {
            resultPromise.then(result => {
              const results = JSON.stringify(['$uuid', null, result])
              $jsChatPanelResponseHandlerInjection
            }).catch(error => {
              const results = JSON.stringify(['$uuid', error.message, null])
              $jsChatPanelResponseHandlerInjection
            })
          } else {
            const results = JSON.stringify(['$uuid', null, resultPromise])
            $jsChatPanelResponseHandlerInjection
          }
        } else {
          const results = JSON.stringify(['$uuid', 'Method not found: $method', null])
          $jsChatPanelResponseHandlerInjection
        }
      })()
    """.trimIndent().trimStart()

    logger.debug("Request to chat panel: $uuid, $method, $paramsJson")
    if (isChatPanelLoaded) {
      executeJs(script)
    } else {
      pendingScripts.add(script)
    }
    return future
  }

  companion object {
    private fun parseVersion(versionString: String): Version? {
      return try {
        val version = versionString.removePrefix("v").substringBefore("-")
        Version.parse(version)
      } catch (e: Exception) {
        null
      }
    }

    private fun checkChatPanelApiVersion(versionString: String): String? {
      val version = parseVersion(versionString)
      val range = Constraint.parse(TABBY_CHAT_PANEL_API_VERSION_RANGE)
      if (version != null && !range.satisfiedBy(version)) {
        return "Please update your Tabby server and Tabby plugin for IntelliJ Platform to the latest version to use chat panel."
      }
      return null
    }

    private fun checkServerHealth(serverHealth: Map<String, Any>?): String? {
      if (serverHealth == null) {
        return "Connecting to Tabby server..."
      }
      if (serverHealth["webserver"] == null || serverHealth["chat_model"] == null) {
        return "You need to launch the server with the chat model enabled; for example, use `--chat-model Qwen2-1.5B-Instruct`."
      }

      if (serverHealth.containsKey("version")) {
        val versionObj = serverHealth["version"]
        val version: Version? = if (versionObj is String) {
          parseVersion(versionObj)
        } else if (versionObj is Map<*, *> && versionObj.containsKey("git_describe")) {
          val gitDescribe = versionObj["git_describe"]
          if (gitDescribe is String) {
            parseVersion(gitDescribe)
          } else {
            null
          }
        } else {
          null
        }
        if (version != null && !Constraint.parse(TABBY_SERVER_VERSION_RANGE).satisfiedBy(version)) {
          return String.format(
            "Tabby Chat requires Tabby server version %s. Your server is running version %s.",
            TABBY_SERVER_VERSION_RANGE, version.toString()
          )
        }
      }
      return null
    }

    private fun escapeCharacters(input: String): String {
      return input.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\"", "\\\"")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace("\b", "\\b")
    }

    private fun positionOneBasedInDocument(document: Document, offset: Int): Position {
      val position = positionInDocument(document, offset)
      return Position(position.line + 1, position.character + 1)
    }

    private fun Color.interpolate(other: Color, fraction: Double): Color {
      val r = (red + (other.red - red) * fraction).toInt().coerceIn(0..255)
      val g = (green + (other.green - green) * fraction).toInt().coerceIn(0..255)
      val b = (blue + (other.blue - blue) * fraction).toInt().coerceIn(0..255)
      return Color(r, g, b)
    }

    private fun Color.toHsl(): String {
      val r = red / 255.0
      val g = green / 255.0
      val b = blue / 255.0
      val max = maxOf(r, g, b)
      val min = minOf(r, g, b)
      var l = (max + min) / 2.0
      var h: Double
      var s: Double

      if (max == min) {
        h = 0.0
        s = 0.0
      } else {
        val delta = max - min
        s = if (l > 0.5) delta / (2.0 - max - min) else delta / (max + min)
        h = when (max) {
          r -> (g - b) / delta + if (g < b) 6 else 0
          g -> (b - r) / delta + 2
          else -> (r - g) / delta + 4
        }
        h /= 6.0
      }

      h *= 360
      s *= 100
      l *= 100

      return String.format("%.0f, %.0f%%, %.0f%%", h, s, l)
    }

    // FIXME: extract this to git provider
    private val gitRemoteUrlToLocalRoot = mutableMapOf<String, String>()

    private fun getDefaultRemoteUrl(repo: GitProvider.Repository): String? {
      if (repo.remotes.isNullOrEmpty()) {
        return null
      }
      val remoteUrl = repo.remotes.firstOrNull { it.name == "origin" }?.url
        ?: repo.remotes.firstOrNull { it.name == "upstream" }?.url
        ?: repo.remotes.firstOrNull()?.url
      if (remoteUrl != null) {
        gitRemoteUrlToLocalRoot[remoteUrl] = repo.root
      }
      return remoteUrl
    }

    private fun String.appendUrlPathSegments(path: String): String {
      return URLBuilder(this).appendPathSegments(path).toString()
    }

    private const val TABBY_CHAT_PANEL_API_VERSION_RANGE = "~0.7.0"
    private const val TABBY_SERVER_VERSION_RANGE = ">=0.25.0"

    private fun loadTabbyThreadsScript(): String {
      val script =
        PluginManagerCore.getPlugin(PluginId.getId("com.tabbyml.intellij-tabby"))
          ?.pluginPath
          ?.resolve("tabby-threads/iife/create-thread-from-iframe.js")
          ?.toFile()
      if (script?.exists() == true) {
        logger<ChatBrowser>().info("Tabby-threads script path: ${script.absolutePath}")
        return script.readText()
      } else {
        throw InitializationException("Tabby-threads script not found. Please reinstall Tabby plugin.")
      }
    }

    private fun loadHtmlContent(script: String) = """
      <!DOCTYPE html>
      <html lang="en">
      
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <style>
          html,
          body,
          div,
          p,
          iframe {
            background: transparent;
            padding: 0;
            margin: 0;
            box-sizing: border-box;
            overflow: hidden;
          }
      
          #message {
            display: flex;
            justify-content: center;
            align-items: flex-start;
          }
      
          #message div {
            width: 100%;
            max-width: 300px;
            color: hsl(var(--foreground));
          }
      
          #message a {
            color: hsl(var(--primary));
          }
      
          #message div {
            margin: 16px;
          }
      
          #message div p {
            margin: 8px 0;
          }
      
          iframe {
            border-width: 0;
            width: 100%;
            height: 100vh;
          }
        </style>
      </head>
      
      <body>
        <div id="message">
          <div>
            <h4>Welcome to Tabby Chat</h4>
            <p id="messageContent"></p>
            <a href="javascript:reload();">Reload</a>
          </div>
        </div>
        <iframe id="chat" style="display: none;"></iframe>
        <script>
        $script
        </script>
        <script>
          function getChatPanel() {
            return document.getElementById("chat");
          }
      
          function reload() {
            // handleReload is a function injected by the client
            handleReload();
          }
      
          function showMessage(message) {
            const messageDiv = document.getElementById("message");
            messageDiv.style.cssText = "display: " + (message ? "flex" : "none") + ";";
            const messageContent = document.getElementById("messageContent");
            messageContent.innerHTML = message;
          }
      
          function showChatPanel(visible) {
            const chat = getChatPanel();
            chat.style.cssText = "display: " + (visible ? "block" : "none") + ";";
          }
      
          function loadChatPanel(url) {
            const chat = getChatPanel();
            chat.src = url;
          }
      
          function applyStyle(style) {
            const { theme, css } = JSON.parse(style);
            document.documentElement.className = theme;
            document.documentElement.style.cssText = css;
          }
      
          window.addEventListener("focus", (event) => {
            const chat = getChatPanel();
            if (chat.style.cssText == "display: block;") {
              setTimeout(() => {
                chat.contentWindow.focus();
              }, 1);
            }
          });
        </script>
      </body>
      
      </html>
    """.trimIndent().trimStart()
  }
}
