package com.tabbyml.intellijtabby.lsp.protocol.client

import com.tabbyml.intellijtabby.lsp.protocol.*
import com.tabbyml.intellijtabby.lsp.protocol.InitializeParams
import com.tabbyml.intellijtabby.lsp.protocol.server.LanguageServer
import org.eclipse.lsp4j.*
import org.eclipse.lsp4j.jsonrpc.services.JsonNotification
import org.eclipse.lsp4j.jsonrpc.services.JsonRequest
import java.util.concurrent.CompletableFuture

abstract class LanguageClient {
  abstract fun buildInitializeParams(): InitializeParams

  open fun processInitializeResult(server: LanguageServer, result: InitializeResult?) {}

  @JsonNotification("tabby/config/didChange")
  open fun configDidChange(params: Config) {
    throw UnsupportedOperationException()
  }

  @JsonNotification("tabby/status/didChange")
  open fun statusDidChange(params: StatusInfo) {
    throw UnsupportedOperationException()
  }

  @JsonRequest("tabby/workspaceFileSystem/readFile")
  open fun readFile(params: ReadFileParams): CompletableFuture<ReadFileResult?> {
    throw UnsupportedOperationException()
  }

  @JsonRequest("tabby/languageSupport/textDocument/declaration")
  open fun declaration(params: DeclarationParams): CompletableFuture<List<LocationLink>?> {
    throw UnsupportedOperationException()
  }

  @JsonRequest("tabby/languageSupport/textDocument/semanticTokens/range")
  open fun semanticTokensRange(params: SemanticTokensRangeParams): CompletableFuture<SemanticTokensRangeResult?> {
    throw UnsupportedOperationException()
  }

  @JsonRequest("tabby/git/repository")
  open fun gitRepository(params: GitRepositoryParams): CompletableFuture<GitRepository?> {
    throw UnsupportedOperationException()
  }

  @JsonRequest("tabby/git/diff")
  open fun gitDiff(params: GitDiffParams): CompletableFuture<GitDiffResult?> {
    throw UnsupportedOperationException()
  }

  @JsonRequest("tabby/editorOptions")
  open fun editorOptions(params: EditorOptionsParams): CompletableFuture<EditorOptions?> {
    throw UnsupportedOperationException()
  }

  @JsonRequest("workspace/applyEdit")
  open fun applyEdit(params: ApplyWorkspaceEditParams): CompletableFuture<ApplyWorkspaceEditResponse> {
    throw UnsupportedOperationException()
  }

  @JsonRequest("client/registerCapability")
  open fun registerCapability(params: RegistrationParams): CompletableFuture<Void> {
    throw UnsupportedOperationException()
  }

  @JsonRequest("client/unregisterCapability")
  open fun unregisterCapability(params: UnregistrationParams): CompletableFuture<Void> {
    throw UnsupportedOperationException()
  }

  @JsonNotification("telemetry/event")
  open fun telemetryEvent(params: Any) {
    throw UnsupportedOperationException()
  }

  @JsonNotification("textDocument/publishDiagnostics")
  open fun publishDiagnostics(params: PublishDiagnosticsParams) {
    throw UnsupportedOperationException()
  }

  @JsonNotification("window/showMessage")
  open fun showMessage(params: MessageParams) {
    throw UnsupportedOperationException()
  }

  @JsonRequest("window/showMessageRequest")
  open fun showMessageRequest(params: ShowMessageRequestParams): CompletableFuture<MessageActionItem?> {
    throw UnsupportedOperationException()
  }

  @JsonRequest("window/showDocument")
  open fun showDocument(params: ShowDocumentParams): CompletableFuture<ShowDocumentResult> {
    throw UnsupportedOperationException()
  }

  @JsonNotification("window/logMessage")
  open fun logMessage(params: MessageParams) {
    throw UnsupportedOperationException()
  }

  @JsonRequest("workspace/workspaceFolders")
  open fun workspaceFolders(): CompletableFuture<List<WorkspaceFolder>> {
    throw UnsupportedOperationException()
  }

  @JsonRequest("workspace/configuration")
  open fun configuration(params: Any): CompletableFuture<List<ClientProvidedConfig>?> {
    throw UnsupportedOperationException()
  }

  @JsonRequest("window/workDoneProgress/create")
  open fun createProgress(params: WorkDoneProgressCreateParams): CompletableFuture<Void> {
    throw UnsupportedOperationException()
  }

  @JsonNotification("$/progress")
  open fun notifyProgress(params: ProgressParams) {
    throw UnsupportedOperationException()
  }

  @JsonNotification("$/logTrace")
  open fun logTrace(params: LogTraceParams) {
    throw UnsupportedOperationException()
  }

  @JsonRequest("workspace/semanticTokens/refresh")
  open fun refreshSemanticTokens(): CompletableFuture<Void> {
    throw UnsupportedOperationException()
  }

  @JsonRequest("workspace/codeLens/refresh")
  open fun refreshCodeLenses(): CompletableFuture<Void> {
    throw UnsupportedOperationException()
  }

  @JsonRequest("workspace/inlayHint/refresh")
  open fun refreshInlayHints(): CompletableFuture<Void> {
    throw UnsupportedOperationException()
  }

  @JsonRequest("workspace/inlineValue/refresh")
  open fun refreshInlineValues(): CompletableFuture<Void> {
    throw UnsupportedOperationException()
  }

  @JsonRequest("workspace/diagnostic/refresh")
  open fun refreshDiagnostics(): CompletableFuture<Void> {
    throw UnsupportedOperationException()
  }

  @JsonRequest("tabby/workspace/applyEdit")
  open fun applyWorkspaceEdit(edit: TabbyApplyWorkspaceEditParams): CompletableFuture<Boolean> {
    throw UnsupportedOperationException()
  }
}
