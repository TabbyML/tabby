package com.tabbyml.intellijtabby.events

import com.intellij.openapi.Disposable
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.event.CaretEvent
import com.intellij.openapi.editor.event.DocumentEvent
import com.intellij.openapi.editor.event.EditorFactoryEvent
import com.intellij.openapi.editor.event.SelectionEvent
import com.tabbyml.intellijtabby.safeSyncPublisher

class EditorFactoryListener : com.intellij.openapi.editor.event.EditorFactoryListener {
  private val logger = Logger.getInstance(EditorFactoryListener::class.java)
  private val listeners = mutableMapOf<Editor, Disposable>()

  override fun editorCreated(event: EditorFactoryEvent) {
    logger.debug("EditorFactoryListener: editorCreated $event")
    val editor = event.editor
    val project = editor.project ?: return
    project.safeSyncPublisher(DocumentListener.TOPIC)?.documentOpened(editor.document, editor)

    val caretListener = object : com.intellij.openapi.editor.event.CaretListener {
      override fun caretPositionChanged(event: CaretEvent) {
        logger.debug("CaretListener: caretPositionChanged $editor $event")
        project.safeSyncPublisher(CaretListener.TOPIC)?.caretPositionChanged(editor, event)
      }
    }

    val selectionListener = object : com.intellij.openapi.editor.event.SelectionListener {
      override fun selectionChanged(event: SelectionEvent) {
        logger.debug("SelectionListener: selectionChanged $editor $event")
        project.safeSyncPublisher(SelectionListener.TOPIC)?.selectionChanged(editor, event)
      }
    }

    val documentListener = object : com.intellij.openapi.editor.event.DocumentListener {
      override fun documentChanged(event: DocumentEvent) {
        logger.debug("DocumentListener: documentChanged $editor $event")
        project.safeSyncPublisher(DocumentListener.TOPIC)?.documentChanged(editor.document, editor, event)
      }
    }

    editor.caretModel.addCaretListener(caretListener)
    editor.selectionModel.addSelectionListener(selectionListener)
    editor.document.addDocumentListener(documentListener)

    listeners[editor] = Disposable {
      editor.caretModel.removeCaretListener(caretListener)
      editor.selectionModel.removeSelectionListener(selectionListener)
      editor.document.removeDocumentListener(documentListener)
    }
  }

  override fun editorReleased(event: EditorFactoryEvent) {
    logger.debug("EditorFactoryListener: editorReleased $event")
    val editor = event.editor
    val project = editor.project ?: return
    project.safeSyncPublisher(DocumentListener.TOPIC)?.documentClosed(editor.document, editor)

    listeners[event.editor]?.dispose()
    listeners.remove(event.editor)
  }
}
