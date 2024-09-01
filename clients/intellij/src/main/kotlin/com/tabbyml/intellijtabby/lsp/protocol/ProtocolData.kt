package com.tabbyml.intellijtabby.lsp.protocol

import com.tabbyml.intellijtabby.lsp.protocol.ClientProvidedConfig.InlineCompletionConfig.TriggerMode
import com.tabbyml.intellijtabby.lsp.protocol.ClientProvidedConfig.Keybindings
import com.tabbyml.intellijtabby.lsp.protocol.EventParams.EventType
import com.tabbyml.intellijtabby.lsp.protocol.EventParams.SelectKind
import com.tabbyml.intellijtabby.lsp.protocol.IssueDetailParams.HelpMessageFormat
import com.tabbyml.intellijtabby.lsp.protocol.ReadFileParams.Format
import org.eclipse.lsp4j.*

data class InitializeParams(
  val processId: Int? = null,
  val clientInfo: ClientInfo? = null,
  val initializationOptions: InitializationOptions? = null,
  val capabilities: ClientCapabilities,
  /**
   * [TraceValue]
   */
  val trace: String? = null,
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
  /**
   * [Keybindings]
   */
  val keybindings: String? = null,
  val anonymousUsageTracking: AnonymousUsageTrackingConfig? = null,
) {
  data class ServerConfig(
    val endpoint: String? = null,
    val token: String? = null,
  )

  data class InlineCompletionConfig(
    /**
     * [TriggerMode]
     */
    val triggerMode: String? = null,
  ) {
    sealed class TriggerMode {
      companion object {
        const val AUTO = "auto"
        const val MANUAL = "manual"
      }
    }
  }

  sealed class Keybindings {
    companion object {
      const val DEFAULT = "default"
      const val TABBY_STYLE = "tabby-style"
      const val CUSTOMIZE = "customize"
    }
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
    enum class InlineCompletionTriggerKind(val value: Int) {
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
  /**
   * [EventType]
   */
  val type: String,
  /**
   * [SelectKind]
   */
  val selectKind: String? = null,
  val eventId: CompletionEventId? = null,
  val viewId: String? = null,
  val elapsed: Int? = null,
) {
  sealed class EventType {
    companion object {
      const val VIEW = "view"
      const val SELECT = "select"
      const val DISMISS = "dismiss"
    }
  }

  sealed class SelectKind {
    companion object {
      const val LINE = "line"
    }
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
  /**
   * [Status]
   */
  val status: String,
)

sealed class Status {
  companion object {
    const val NOT_INITIALIZED = "notInitialized"
    const val READY = "ready"
    const val DISCONNECTED = "disconnected"
    const val UNAUTHORIZED = "unauthorized"
    const val FINALIZED = "finalized"
  }
}

typealias DidUpdateIssueParams = IssueList

data class IssueList(
  /**
   * List of [IssueName]
   */
  val issues: List<String>
)

sealed class IssueName {
  companion object {
    const val SLOW_COMPLETION_RESPONSE_TIME = "slowCompletionResponseTime"
    const val HIGH_COMPLETION_TIMEOUT_RATE = "highCompletionTimeoutRate"
    const val CONNECTION_FAILED = "connectionFailed"
  }
}

data class IssueDetailParams(
  /**
   * [IssueName]
   */
  val name: String,
  /**
   * [HelpMessageFormat]
   */
  val helpMessageFormat: String? = null,
) {
  sealed class HelpMessageFormat {
    companion object {
      const val MARKDOWN = "markdown"
      const val HTML = "html"
    }
  }
}

data class IssueDetailResult(
  /**
   * [IssueName]
   */
  val name: String,
  val helpMessage: String? = null,
)

data class ReadFileParams(
  val uri: String,
  /**
   * [Format]
   */
  val format: String,
  val range: Range? = null,
) {
  sealed class Format {
    companion object {
      const val TEXT = "text"
    }
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
