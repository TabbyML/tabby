package com.tabbyml.intellijtabby.lsp.protocol.server

import com.tabbyml.intellijtabby.lsp.protocol.DidChangeActiveEditorParams
import org.eclipse.lsp4j.jsonrpc.services.JsonNotification
import org.eclipse.lsp4j.jsonrpc.services.JsonSegment

@JsonSegment("tabby/editors")
interface EditorsFeature {
  @JsonNotification
  fun didChangeActiveEditor(params: DidChangeActiveEditorParams)
}
