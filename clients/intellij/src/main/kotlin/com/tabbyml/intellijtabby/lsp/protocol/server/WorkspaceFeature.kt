package com.tabbyml.intellijtabby.lsp.protocol.server

import com.tabbyml.intellijtabby.lsp.protocol.DidChangeConfigurationParams
import org.eclipse.lsp4j.DidChangeWorkspaceFoldersParams
import org.eclipse.lsp4j.ExecuteCommandParams
import org.eclipse.lsp4j.jsonrpc.services.JsonNotification
import org.eclipse.lsp4j.jsonrpc.services.JsonRequest
import org.eclipse.lsp4j.jsonrpc.services.JsonSegment
import java.util.concurrent.CompletableFuture

@JsonSegment("workspace")
interface WorkspaceFeature {
  @JsonNotification
  fun didChangeConfiguration(params: DidChangeConfigurationParams)

  @JsonNotification
  fun didChangeWorkspaceFolders(params: DidChangeWorkspaceFoldersParams)

  @JsonRequest
  fun executeCommand(params: ExecuteCommandParams): CompletableFuture<Any?>
}
