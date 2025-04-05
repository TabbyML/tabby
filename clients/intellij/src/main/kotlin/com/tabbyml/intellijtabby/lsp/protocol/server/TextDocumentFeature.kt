package com.tabbyml.intellijtabby.lsp.protocol.server

import com.tabbyml.intellijtabby.lsp.protocol.CompletionList
import com.tabbyml.intellijtabby.lsp.protocol.InlineCompletionList
import com.tabbyml.intellijtabby.lsp.protocol.InlineCompletionParams
import org.eclipse.lsp4j.*
import org.eclipse.lsp4j.jsonrpc.services.JsonNotification
import org.eclipse.lsp4j.jsonrpc.services.JsonRequest
import org.eclipse.lsp4j.jsonrpc.services.JsonSegment
import java.util.concurrent.CompletableFuture

@JsonSegment("textDocument")
interface TextDocumentFeature {
  @JsonRequest
  fun completion(params: CompletionParams): CompletableFuture<CompletionList?>

  @JsonRequest
  fun inlineCompletion(params: InlineCompletionParams): CompletableFuture<InlineCompletionList?>

  @JsonNotification
  fun didOpen(params: DidOpenTextDocumentParams)

  @JsonNotification
  fun didChange(params: DidChangeTextDocumentParams)

  @JsonNotification
  fun didClose(params: DidCloseTextDocumentParams)

  @JsonNotification
  fun didSave(params: DidSaveTextDocumentParams)

  @JsonNotification
  fun willSave(params: WillSaveTextDocumentParams)

  @JsonRequest
  fun codeLens(params: CodeLensParams): CompletableFuture<List<CodeLens>?>
}
