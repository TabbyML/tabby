package com.tabbyml.intellijtabby.lsp.protocol.server

import com.tabbyml.intellijtabby.lsp.protocol.DidChangeConfigurationParams
import org.eclipse.lsp4j.DidChangeWorkspaceFoldersParams
import org.eclipse.lsp4j.jsonrpc.services.JsonNotification
import org.eclipse.lsp4j.jsonrpc.services.JsonSegment

@JsonSegment("workspace")
interface WorkspaceFeature {
  @JsonNotification
  fun didChangeConfiguration(params: DidChangeConfigurationParams)

  @JsonNotification
  fun didChangeWorkspaceFolders(params: DidChangeWorkspaceFoldersParams)
}
