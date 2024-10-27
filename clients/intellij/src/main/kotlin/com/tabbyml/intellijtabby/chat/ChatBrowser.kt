package com.tabbyml.intellijtabby.chat

import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import com.google.gson.reflect.TypeToken
import com.intellij.openapi.application.ReadAction
import com.intellij.openapi.application.invokeLater
import com.intellij.openapi.command.WriteCommandAction
import com.intellij.openapi.components.service
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.editor.colors.EditorColors
import com.intellij.openapi.editor.colors.EditorColorsListener
import com.intellij.openapi.editor.colors.EditorColorsManager
import com.intellij.openapi.editor.colors.EditorFontType
import com.intellij.openapi.fileEditor.FileEditorManager
import com.intellij.openapi.progress.util.BackgroundTaskUtil
import com.intellij.openapi.project.Project
import com.intellij.openapi.project.guessProjectDir
import com.intellij.openapi.util.SystemInfo
import com.intellij.ui.jcef.JBCefBrowser
import com.intellij.ui.jcef.JBCefBrowserBase
import com.intellij.ui.jcef.JBCefJSQuery
import com.tabbyml.intellijtabby.events.CombinedState
import com.tabbyml.intellijtabby.git.GitProvider
import com.tabbyml.intellijtabby.lsp.protocol.ServerInfo
import com.tabbyml.intellijtabby.lsp.protocol.Status
import io.github.z4kn4fein.semver.Version
import io.github.z4kn4fein.semver.constraints.Constraint
import io.github.z4kn4fein.semver.constraints.satisfiedBy
import org.cef.browser.CefBrowser
import org.cef.handler.CefLoadHandlerAdapter
import java.awt.Color
import java.awt.Toolkit
import java.awt.datatransfer.StringSelection
import java.io.File


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
  private val logger = Logger.getInstance(ChatBrowser::class.java)
  private val gson = Gson()
  private val combinedState = project.service<CombinedState>()
  private val gitProvider = project.service<GitProvider>()
  private val messageBusConnection = project.messageBus.connect()

  private val reloadHandler = JBCefJSQuery.create(this as JBCefBrowserBase)
  private val chatPanelRequestHandler = JBCefJSQuery.create(this as JBCefBrowserBase)

  private var currentConfig: ServerInfo.ServerInfoConfig? = null
  private var isChatPanelLoaded = false
  private val pendingScripts: MutableList<String> = mutableListOf()

  private data class ChatPanelRequest(
    val method: String,
    val params: List<Any?>,
  )

  private data class FileContext(
    val kind: String = "file",
    val range: LineRange,
    val filepath: String,
    val content: String,
    @SerializedName("git_url")
    val gitUrl: String,
  ) {
    data class LineRange(
      val start: Int,
      val end: Int,
    )
  }

  init {
    component.isVisible = false
    val bgColor = calcComponentBgColor()
    component.background = bgColor
    setPageBackgroundColor("hsl(${bgColor.toHsl()})")

    loadHTML(HTML_CONTENT)

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

    reloadHandler.addHandler {
      reloadContent(true)
      return@addHandler JBCefJSQuery.Response("")
    }

    chatPanelRequestHandler.addHandler { message ->
      val request = gson.fromJson(message, ChatPanelRequest::class.java)
      handleChatPanelRequest(request)
      return@addHandler JBCefJSQuery.Response("")
    }

    messageBusConnection.subscribe(CombinedState.Listener.TOPIC, object : CombinedState.Listener {
      override fun stateChanged(state: CombinedState.State) {
        reloadContent()
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
      val context = getActiveFileContext()
      chatPanelSendMessage(PROMPT_EXPLAIN, context)
    }
  }

  fun fixSelectedText() {
    BackgroundTaskUtil.executeOnPooledThread(this) {
      // FIXME(@icycodes): collect the diagnostic message provided by IDE
      val context = getActiveFileContext()
      chatPanelSendMessage(PROMPT_FIX, context)
    }
  }

  fun generateDocsForSelectedText() {
    BackgroundTaskUtil.executeOnPooledThread(this) {
      val context = getActiveFileContext()
      chatPanelSendMessage(PROMPT_GENERATE_DOCS, context)
    }
  }

  fun generateTestsForSelectedText() {
    BackgroundTaskUtil.executeOnPooledThread(this) {
      val context = getActiveFileContext()
      chatPanelSendMessage(PROMPT_GENERATE_TESTS, context)
    }
  }

  fun addActiveEditorAsContext(useSelectedText: Boolean) {
    BackgroundTaskUtil.executeOnPooledThread(this) {
      val context = getActiveFileContext(useSelectedText) ?: return@executeOnPooledThread
      chatPanelAddRelevantContext(context)
    }
  }

  private fun getActiveFileContext(useSelectedText: Boolean = true): FileContext? {
    return FileEditorManager.getInstance(project).selectedTextEditor?.let { editor ->
      ReadAction.compute<Triple<String, Int, Int>?, Throwable> {
        val document = editor.document
        if (useSelectedText) {
          val selectionModel = editor.selectionModel
          val text = selectionModel.selectedText.takeUnless { it.isNullOrBlank() } ?: return@compute null
          Triple(
            text,
            document.getLineNumber(selectionModel.selectionStart) + 1,
            document.getLineNumber(selectionModel.selectionEnd) + 1,
          )
        } else {
          val text = document.text.takeUnless { it.isBlank() } ?: return@compute null
          Triple(
            text,
            1,
            document.lineCount,
          )
        }
      }?.let { context ->
        val uri = editor.virtualFile?.url
        val gitRepo = uri?.let { gitProvider.getRepository(it) }
        val relativeBase = gitRepo?.root ?: project.guessProjectDir()?.url
        val relativePath = uri?.let {
          if (!relativeBase.isNullOrBlank() && it.startsWith(relativeBase)) {
            it.substringAfter(relativeBase).trimStart(File.separatorChar)
          } else it
        }
        logger.debug("Active context: context: $context, uri: $uri, gitRepo: $gitRepo, relativePath: $relativePath, relativeBase: $relativeBase")

        FileContext(
          range = FileContext.LineRange(
            start = context.second,
            end = context.third,
          ),
          filepath = relativePath ?: "",
          content = context.first,
          gitUrl = gitRepo?.remotes?.firstOrNull()?.url ?: "",
        )
      }
    }
  }

  private fun handleLoaded() {
    jsInjectHandlers()
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
    val borderColor = editorColorsScheme.getColor(EditorColors.BORDER_LINES_COLOR)
      ?: if (isDarkTheme) editorColorsScheme.defaultForeground.brighter() else editorColorsScheme.defaultForeground.darker()
    val primaryColor = editorColorsScheme.getAttributes(EditorColors.REFERENCE_HYPERLINK_COLOR).foregroundColor
    val font = editorColorsScheme.getFont(EditorFontType.PLAIN).fontName
    val fontSize = editorColorsScheme.editorFontSize
    val css = String.format("background-color: hsl(%s);", bgActiveColor.toHsl()) +
        String.format("--background: %s;", bgColor.toHsl()) +
        String.format("--foreground: %s;", fgColor.toHsl()) +
        String.format("--border: %s;", borderColor.toHsl()) +
        String.format("--primary: %s;", primaryColor.toHsl()) +
        String.format("font: %s;", font) +
        String.format("font-size: %spx;", fontSize) +
        // FIXME(@icycodes): remove these once the server no longer reads the '--intellij-editor' css vars
        String.format("--intellij-editor-background: %s;", bgColor.toHsl()) +
        String.format("--intellij-editor-foreground: %s;", fgColor.toHsl()) +
        String.format("--intellij-editor-border: %s;", borderColor.toHsl())
    logger.debug("CSS: $css")
    return css
  }

  private fun reloadContent(force: Boolean = false) {
    if (force) {
      // FIXME(@icycodes): force reload requires await reconnection then get server health
      reloadContentInternal(true)
    } else {
      reloadContentInternal(false)
    }
  }

  private fun reloadContentInternal(force: Boolean = false) {
    val status = combinedState.state.agentStatus
    when (status) {
      Status.NOT_INITIALIZED, Status.FINALIZED -> {
        showContent("Initializing...")
      }

      Status.DISCONNECTED -> {
        showContent("Cannot connect to Tabby server, please check your settings.")
      }

      Status.UNAUTHORIZED -> {
        showContent("Authorization required, please set your token in settings.")
      }

      else -> {
        val health = combinedState.state.agentServerInfo?.health
        val error = checkServerHealth(health)
        if (error != null) {
          showContent(error)
        } else {
          val config = combinedState.state.agentServerInfo?.config
          if (config == null) {
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

  private fun showContent(message: String? = null) {
    if (message != null) {
      jsShowMessage(message)
      jsShowChatPanel(false)
    } else {
      jsShowMessage(null)
      jsShowChatPanel(true)
    }
  }

  private fun handleChatPanelRequest(request: ChatPanelRequest) {
    when (request.method) {
      "navigate" -> {
        logger.debug("navigate: request: ${request.params}")
        // FIXME(@icycodes): not implemented yet
      }

      "refresh" -> {
        logger.debug("refresh: request: ${request.params}")
        reloadContent(true)
      }

      "onSubmitMessage" -> {
        logger.debug("onSubmitMessage: request: ${request.params}")
        if (request.params.isNotEmpty()) {
          val message = request.params[0] as String
          val relevantContext: List<FileContext>? = request.params.getOrNull(1)?.let {
            gson.fromJson(gson.toJson(it), object : TypeToken<List<FileContext?>?>() {}.type)
          }
          val activeContext = getActiveFileContext()
          chatPanelSendMessage(message, null, relevantContext, activeContext)
        }
      }

      "onApplyInEditor" -> {
        logger.debug("onApplyInEditor: request: ${request.params}")
        val content = request.params.getOrNull(0) as String? ?: return
        val editor = FileEditorManager.getInstance(project).selectedTextEditor ?: return
        invokeLater {
          WriteCommandAction.runWriteCommandAction(project) {
            val start = editor.selectionModel.selectionStart
            val end = editor.selectionModel.selectionEnd
            editor.document.replaceString(start, end, content)
            editor.caretModel.moveToOffset(start + content.length)
          }
        }
      }

      "onLoaded" -> {
        logger.debug("onLoaded: request: ${request.params}")
        val params = request.params.getOrNull(0) as Map<*, *>?
        val apiVersion = params?.get("apiVersion") as String?
        if (apiVersion != null) {
          val error = checkChatPanelApiVersion(apiVersion)
          if (error != null) {
            showContent(error)
            return
          }
        }
        isChatPanelLoaded = true
        chatPanelInit()
        chatPanelUpdateTheme()
        showContent()
        pendingScripts.forEach { executeJs(it) }
        pendingScripts.clear()
      }

      "onCopy" -> {
        logger.debug("onCopy: request: ${request.params}")
        val content = request.params.getOrNull(0) as String? ?: return
        val stringSelection = StringSelection(content)
        val clipboard = Toolkit.getDefaultToolkit().systemClipboard
        clipboard.setContents(stringSelection, null)
      }

      "onKeyboardEvent" -> {
        // nothing to do
      }
    }
  }

  // chat panel api functions

  private fun chatPanelInit() {
    val request = ChatPanelRequest(
      "init",
      listOf(
        mapOf(
          "fetcherOptions" to mapOf(
            "authorization" to currentConfig?.token,
          )
        )
      )
    )
    logger.debug("chatPanelInit: $request")
    jsSendRequestToChatPanel(request)
  }

  private fun chatPanelSendMessage(
    message: String,
    selectContext: FileContext? = null,
    relevantContext: List<FileContext>? = null,
    activeContext: FileContext? = null,
  ) {
    val request = ChatPanelRequest(
      "sendMessage",
      listOf(
        mapOf(
          "message" to message,
          "selectContext" to selectContext,
          "relevantContext" to relevantContext,
          "activeContext" to activeContext,
        )
      )
    )
    logger.debug("chatPanelSendMessage: $request")
    jsSendRequestToChatPanel(request)
  }

  private fun chatPanelAddRelevantContext(context: FileContext) {
    val request = ChatPanelRequest(
      "addRelevantContext",
      listOf(context)
    )
    logger.debug("chatPanelAddRelevantContext: $request")
    jsSendRequestToChatPanel(request)
  }

  private fun chatPanelUpdateTheme() {
    val request = ChatPanelRequest(
      "updateTheme",
      listOf(
        buildCss(),
        if (isDarkTheme) "dark" else "light",
      )
    )
    logger.debug("chatPanelUpdateTheme: $request")
    jsSendRequestToChatPanel(request)
  }

  private fun chatPanelUpdateActiveSelection(context: FileContext?) {
    val request = ChatPanelRequest(
      "updateActiveSelection",
      listOf(context)
    )
    logger.debug("chatPanelUpdateActiveSelection: $request")
    jsSendRequestToChatPanel(request)
  }

  // js functions

  private fun jsInjectHandlers() {
    val script = String.format(
      """
        window.handleReload = function() { %s }
        window.handleChatPanelRequest = function(message) { %s }
      """.trimIndent(),
      reloadHandler.inject(""),
      chatPanelRequestHandler.inject("message"),
    )
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
    val chatUrl = String.format("%s/chat?client=intellij", config.endpoint)
    val script = String.format("loadChatPanel('%s')", chatUrl)
    executeJs(script)
  }

  private fun jsSendRequestToChatPanel(request: ChatPanelRequest) {
    val json = gson.toJson(request)
    val script = String.format("sendRequestToChatPanel('%s')", escapeCharacters(json))
    if (isChatPanelLoaded) {
      executeJs(script)
    } else {
      pendingScripts.add(script)
    }
  }

  private fun executeJs(script: String) {
    cefBrowser.executeJavaScript(script, cefBrowser.url, 0)
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

    private const val TABBY_CHAT_PANEL_API_VERSION_RANGE = "~0.2.0"
    private const val TABBY_SERVER_VERSION_RANGE = ">=0.18.0"

    private const val PROMPT_EXPLAIN: String = "Explain the selected code:"
    private const val PROMPT_FIX: String = "Identify and fix potential bugs in the selected code:"
    private const val PROMPT_GENERATE_DOCS: String = "Generate documentation for the selected code:"
    private const val PROMPT_GENERATE_TESTS: String = "Generate a unit test for the selected code:"

    private const val HTML_CONTENT = """
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
      
          function sendRequestToChatPanel(request) {
            const chat = getChatPanel();
            // client to server requests
            const { method, params } = JSON.parse(request);
            if (method) {
              // adapter for @quilted/threads requests
              const data = [
                0, // kind: Request
                [uuid(), method, params],
              ]
              chat.contentWindow.postMessage(data, new URL(chat.src).origin);
            }
          }
      
          function uuid() {
            return ([1e7] + -1e3 + -4e3 + -8e3 + -1e11).replace(/[018]/g, c =>
              (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
            );
          }
      
          window.addEventListener("focus", (event) => {
            const chat = getChatPanel();
            if (chat.style.cssText == "display: block;") {
              setTimeout(() => {
                chat.contentWindow.focus();
              }, 1);
            }
          });

          window.addEventListener("message", (event) => {
            const chat = getChatPanel();
      
            // server to client requests
            if (event.source === chat.contentWindow) {      
              // adapter for @quilted/threads requests
              if (Array.isArray(event.data) && event.data.length >= 2) {
                const [kind, data] = event.data;
                if (kind === 0) {
                  // 0: Request
                  if (Array.isArray(data) && event.data.length >= 2) {
                    const [_requestId, method, params] = data;
                    // handleChatPanelRequest is a function injected by the client
                    handleChatPanelRequest(JSON.stringify({ method, params }));
                  }
                } else {
                  // 1: Response
                  // ignored as current methods return void
                }
              }
            }
          });
        </script>
      </body>
      
      </html>
    """
  }
}
