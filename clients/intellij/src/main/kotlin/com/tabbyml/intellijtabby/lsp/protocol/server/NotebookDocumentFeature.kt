package com.tabbyml.intellijtabby.lsp.protocol.server

import org.eclipse.lsp4j.DidChangeNotebookDocumentParams
import org.eclipse.lsp4j.DidCloseNotebookDocumentParams
import org.eclipse.lsp4j.DidOpenNotebookDocumentParams
import org.eclipse.lsp4j.DidSaveNotebookDocumentParams
import org.eclipse.lsp4j.jsonrpc.services.JsonNotification
import org.eclipse.lsp4j.jsonrpc.services.JsonSegment

@JsonSegment("notebookDocument")
interface NotebookDocumentFeature {
  fun didOpen(params: DidOpenNotebookDocumentParams)

  @JsonNotification
  fun didChange(params: DidChangeNotebookDocumentParams)

  @JsonNotification
  fun didSave(params: DidSaveNotebookDocumentParams)

  @JsonNotification
  fun didClose(params: DidCloseNotebookDocumentParams)
}
