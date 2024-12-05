package com.tabbyml.intellijtabby.lsp

import com.intellij.openapi.Disposable
import com.intellij.openapi.editor.Document
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.EditorFactory
import com.intellij.openapi.editor.event.DocumentEvent
import com.intellij.openapi.project.Project
import com.tabbyml.intellijtabby.events.DocumentListener
import com.tabbyml.intellijtabby.findPsiFile
import com.tabbyml.intellijtabby.lsp.protocol.server.LanguageServer
import org.eclipse.lsp4j.*

class TextDocumentSync(private val project: Project) : Disposable {
  private val messageBusConnection = project.messageBus.connect()

  fun startSync(server: LanguageServer) {
    initSync(server)
    registerListeners(server)
  }

  private fun initSync(server: LanguageServer) {
    val editorFactory = EditorFactory.getInstance()
    editorFactory.allEditors.forEach { editor ->
      buildDidOpenTextDocumentParams(editor)?.let { server.textDocumentFeature.didOpen(it) }
    }
  }

  private fun registerListeners(server: LanguageServer) {
    messageBusConnection.subscribe(DocumentListener.TOPIC, object : DocumentListener {
      override fun documentOpened(document: Document, editor: Editor) {
        buildDidOpenTextDocumentParams(editor)?.let { server.textDocumentFeature.didOpen(it) }
      }

      override fun documentClosed(document: Document, editor: Editor) {
        buildDidCloseTextDocumentParams(editor)?.let { server.textDocumentFeature.didClose(it) }
      }

      override fun documentChanged(document: Document, editor: Editor, event: DocumentEvent) {
        buildDidChangeTextDocumentParams(editor, event)?.let { server.textDocumentFeature.didChange(it) }
      }
    })
  }

  override fun dispose() {
    messageBusConnection.dispose()
  }

  companion object {
    fun buildDidOpenTextDocumentParams(editor: Editor): DidOpenTextDocumentParams? {
      val project = editor.project ?: return null
      val virtualFile = editor.virtualFile ?: return null
      return DidOpenTextDocumentParams(
        TextDocumentItem(
          virtualFile.url,
          project.findPsiFile(virtualFile)?.getLanguageId() ?: "plaintext",
          editor.document.modificationStamp.toInt(),
          editor.document.text,
        )
      )
    }

    fun buildDidCloseTextDocumentParams(editor: Editor): DidCloseTextDocumentParams? {
      val virtualFile = editor.virtualFile ?: return null
      return DidCloseTextDocumentParams(
        TextDocumentIdentifier(virtualFile.url)
      )
    }

    fun buildDidChangeTextDocumentParams(editor: Editor, event: DocumentEvent): DidChangeTextDocumentParams? {
      val virtualFile = editor.virtualFile ?: return null
      val oldLines = event.oldFragment.lines()
      val startPosition = positionInDocument(editor.document, event.offset)
      val endPosition = Position(
        startPosition.line + oldLines.size - 1,
        (if (oldLines.size == 1) startPosition.character else 0) + oldLines.last().length,
      )
      return DidChangeTextDocumentParams(
        VersionedTextDocumentIdentifier(
          virtualFile.url,
          editor.document.modificationStamp.toInt(),
        ), listOf(
          TextDocumentContentChangeEvent(
            Range(startPosition, endPosition), event.newFragment.toString()
          )
        )
      )

    }
  }
}
