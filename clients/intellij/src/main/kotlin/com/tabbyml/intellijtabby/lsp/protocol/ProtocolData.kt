package com.tabbyml.intellijtabby.lsp.protocol

import com.google.gson.annotations.JsonAdapter
import com.google.gson.annotations.SerializedName
import org.eclipse.lsp4j.*

data class InitializeParams(
  val processId: Int? = null,
  val clientInfo: ClientInfo? = null,
  val initializationOptions: InitializationOptions? = null,
  val capabilities: ClientCapabilities,
  val trace: TraceValue? = null,
  val workspaceFolders: List<WorkspaceFolder>? = null
)

data class ClientInfo(
  val name: String,
  val version: String? = null,
  val tabbyPlugin: TabbyPluginInfo? = null,
) {
  data class TabbyPluginInfo(
    val name: String,
    val version: String? = null,
  )
}

data class InitializationOptions(
  val config: ClientProvidedConfig? = null
)

data class ClientCapabilities(
  val workspace: WorkspaceClientCapabilities? = null,
  val textDocument: TextDocumentClientCapabilities? = null,
  val notebookDocument: NotebookDocumentClientCapabilities? = null,
  val tabby: TabbyClientCapabilities? = null,
)


data class TextDocumentClientCapabilities(
  val synchronization: SynchronizationCapabilities? = null,
  val completion: CompletionCapabilities? = null,
  val inlineCompletion: InlineCompletionCapabilities? = null,
)

typealias InlineCompletionCapabilities = DynamicRegistrationCapabilities

data class TabbyClientCapabilities(
  val agent: Boolean? = null,
  val workspaceFileSystem: Boolean? = null,
  val dataStore: Boolean? = null,
  val languageSupport: Boolean? = null,
  val gitProvider: Boolean? = null,
  val editorOptions: Boolean? = null,
)

enum class TraceValue {
  @SerializedName("off")
  OFF,

  @SerializedName("messages")
  MESSAGES,

  @SerializedName("verbose")
  VERBOSE,
}

data class InitializeResult(
  val capabilities: ServerCapabilities,
  val serverInfo: ServerInfo? = null,
) {
  data class ServerInfo(
    val name: String,
    val version: String? = null,
  )
}

data class ServerCapabilities(
  val workspace: WorkspaceServerCapabilities? = null,
  val textDocumentSync: TextDocumentSyncOptions? = null,
  val notebookDocumentSync: NotebookDocumentSyncOptions? = null,
  val completionProvider: CompletionOptions? = null,
  val inlineCompletionProvider: Boolean? = null,
  val tabby: TabbyServerCapabilities? = null,
)

data class TabbyServerCapabilities(
  val chat: Boolean? = null
)

data class ClientProvidedConfig(
  val server: ServerConfig? = null,
  val inlineCompletion: InlineCompletionConfig? = null,
  val keybindings: KeybindingsConfig? = null,
  val anonymousUsageTracking: AnonymousUsageTrackingConfig? = null,
) {
  data class ServerConfig(
    val endpoint: String? = null,
    val token: String? = null,
  )

  data class InlineCompletionConfig(
    val triggerMode: TriggerMode? = null,
  ) {
    enum class TriggerMode {
      @SerializedName("auto")
      AUTO,

      @SerializedName("manual")
      MANUAL,
    }
  }

  enum class KeybindingsConfig {
    @SerializedName("default")
    DEFAULT,

    @SerializedName("tabby-style")
    TABBY_STYLE,

    @SerializedName("customize")
    CUSTOMIZE,
  }

  data class AnonymousUsageTrackingConfig(
    val disable: Boolean? = null,
  )
}

data class DidChangeConfigurationParams(
  val settings: ClientProvidedConfig? = null,
)

data class CompletionList(
  val isIncomplete: Boolean,
  val items: List<CompletionItem>,
)

data class CompletionItem(
  val label: String,
  val labelDetails: CompletionItemLabelDetails? = null,
  val kind: CompletionItemKind? = null,
  val tags: List<CompletionItemTag>? = null,
  val detail: String? = null,
  val documentation: MarkupContent? = null,
  val preselect: Boolean? = null,
  val sortText: String? = null,
  val filterText: String? = null,
  val insertText: String? = null,
  val insertTextFormat: InsertTextFormat? = null,
  val insertTextMode: InsertTextMode? = null,
  val textEdit: TextEdit? = null,
  val textEditText: String? = null,
  val additionalTextEdits: List<TextEdit>? = null,
  val commitCharacters: List<String>? = null,
  val command: Command? = null,
  val data: Data? = null,
) {
  data class Data(
    val eventId: CompletionEventId? = null
  )
}

data class InlineCompletionParams(
  val context: InlineCompletionContext? = null,
  val textDocument: TextDocumentIdentifier,
  val position: Position,
) {
  data class InlineCompletionContext(
    val triggerKind: InlineCompletionTriggerKind,
    val selectedCompletionInfo: SelectedCompletionInfo? = null,
  ) {
    @JsonAdapter(EnumIntTypeAdapter.Factory::class)
    enum class InlineCompletionTriggerKind(override val value: Int) : EnumInt {
      Invoked(0), Automatic(1);
    }

    data class SelectedCompletionInfo(
      val text: String,
      val range: Range,
    )
  }
}

data class InlineCompletionList(
  val isIncomplete: Boolean,
  val items: List<InlineCompletionItem>,
)

data class InlineCompletionItem(
  val insertText: String,
  val filterText: String? = null,
  val range: Range? = null,
  val command: Command? = null,
  val data: Data? = null,
) {

  data class Data(
    val eventId: CompletionEventId? = null
  )
}

data class CompletionEventId(
  val completionId: String,
  val choiceIndex: Int,
)

data class EventParams(
  val type: EventType,
  val selectKind: SelectKind? = null,
  val eventId: CompletionEventId? = null,
  val viewId: String? = null,
  val elapsed: Int? = null,
) {
  enum class EventType {
    @SerializedName("view")
    VIEW,

    @SerializedName("select")
    SELECT,

    @SerializedName("dismiss")
    DISMISS,
  }

  enum class SelectKind {
    @SerializedName("line")
    LINE,
  }
}

data class DidUpdateServerInfoParams(
  val serverInfo: ServerInfo
)

data class ServerInfo(
  val config: ServerInfoConfig,
  val health: Map<String, Any>?,
) {
  data class ServerInfoConfig(
    val endpoint: String,
    val token: String?,
    val requestHeaders: Map<String, Any>?,
  )
}

data class DidChangeStatusParams(
  val status: Status,
)

enum class Status {
  @SerializedName("notInitialized")
  NOT_INITIALIZED,

  @SerializedName("ready")
  READY,

  @SerializedName("disconnected")
  DISCONNECTED,

  @SerializedName("unauthorized")
  UNAUTHORIZED,

  @SerializedName("finalized")
  FINALIZED
}

typealias DidUpdateIssueParams = IssueList

data class IssueList(
  val issues: List<IssueName>
)

enum class IssueName {
  @SerializedName("slowCompletionResponseTime")
  SLOW_COMPLETION_RESPONSE_TIME,

  @SerializedName("highCompletionTimeoutRate")
  HIGH_COMPLETION_TIMEOUT_RATE,

  @SerializedName("connectionFailed")
  CONNECTION_FAILED
}

data class IssueDetailParams(
  val name: IssueName,
  val helpMessageFormat: HelpMessageFormat? = null,
) {
  enum class HelpMessageFormat {
    @SerializedName("markdown")
    MARKDOWN,

    @SerializedName("html")
    HTML,
  }
}

data class IssueDetailResult(
  val name: IssueName,
  val helpMessage: String? = null,
)

data class ReadFileParams(
  val uri: String,
  val format: Format,
  val range: Range? = null,
) {
  enum class Format {
    @SerializedName("text")
    TEXT,
  }
}

data class ReadFileResult(
  val text: String? = null
)

data class DataStoreGetParams(
  val key: String
)

data class DataStoreSetParams(
  val key: String, val value: Any? = null
)

data class SemanticTokensRangeResult(
  val legend: SemanticTokensLegend,
  val tokens: SemanticTokens,
) {
  data class SemanticTokensLegend(
    val tokenTypes: List<String>,
    val tokenModifiers: List<String>,
  )

  data class SemanticTokens(
    val resultId: String,
    val data: List<Int>,
  )
}

data class GitRepositoryParams(
  val uri: String
)

data class GitRepository(
  val root: String,
  val remoteUrl: String? = null,
  val remotes: List<Remote>? = null,
) {
  data class Remote(
    val name: String,
    val url: String,
  )
}

data class GitDiffParams(
  val repository: String,
  val cached: Boolean,
)

data class GitDiffResult(
  val diff: List<String>,
)

data class EditorOptionsParams(
  val uri: String,
)

data class EditorOptions(
  val indentation: String? = null,
)
