package com.tabbyml.intellijtabby.lsp.protocol.server

import com.tabbyml.intellijtabby.lsp.protocol.InitializeParams
import org.eclipse.lsp4j.InitializeResult
import org.eclipse.lsp4j.InitializedParams
import org.eclipse.lsp4j.SetTraceParams
import org.eclipse.lsp4j.jsonrpc.services.JsonDelegate
import org.eclipse.lsp4j.jsonrpc.services.JsonNotification
import org.eclipse.lsp4j.jsonrpc.services.JsonRequest
import java.util.concurrent.CompletableFuture

interface LanguageServer {
  @JsonRequest
  fun initialize(params: InitializeParams): CompletableFuture<InitializeResult?>

  @JsonNotification
  fun initialized(params: InitializedParams)

  @JsonRequest
  fun shutdown(): CompletableFuture<Any?>

  @JsonNotification
  fun exit()

  @get:JsonDelegate
  val notebookDocumentFeature: NotebookDocumentFeature

  @get:JsonDelegate
  val textDocumentFeature: TextDocumentFeature

  @get:JsonDelegate
  val workspaceFeature: WorkspaceFeature

  @get:JsonDelegate
  val telemetryFeature: TelemetryFeature

  @get:JsonDelegate
  val configFeature: ConfigFeature

  @get:JsonDelegate
  val statusFeature: StatusFeature

  @get:JsonDelegate
  val chatFeature: ChatFeature

  @get:JsonDelegate
  val editorsFeature: EditorsFeature

  @JsonNotification("$/setTrace")
  fun setTrace(params: SetTraceParams)
}
