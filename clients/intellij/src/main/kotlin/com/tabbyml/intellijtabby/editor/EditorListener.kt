package com.tabbyml.intellijtabby.editor

import com.intellij.openapi.components.service
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.event.*
import com.intellij.openapi.fileEditor.FileEditorManager
import com.intellij.openapi.fileEditor.FileEditorManagerEvent
import com.intellij.openapi.fileEditor.FileEditorManagerListener
import com.intellij.util.messages.MessageBusConnection

class EditorListener : EditorFactoryListener {
  private val logger = Logger.getInstance(EditorListener::class.java)
  private val messagesConnection = mutableMapOf<Editor, MessageBusConnection>()

  override fun editorCreated(event: EditorFactoryEvent) {
    val editor = event.editor
    val editorManager = editor.project?.let { FileEditorManager.getInstance(it) } ?: return
    val completionScheduler = service<CompletionScheduler>()

    editor.caretModel.addCaretListener(object : CaretListener {
      override fun caretPositionChanged(event: CaretEvent) {
        if (editorManager.selectedTextEditor == editor) {
          completionScheduler.scheduled?.let {
            if (it.editor != editor || it.offset != editor.caretModel.primaryCaret.offset) {
              completionScheduler.clear()
            }
          }
        }
      }
    })

    editor.document.addDocumentListener(object : DocumentListener {
      override fun documentChanged(event: DocumentEvent) {
        if (editorManager.selectedTextEditor == editor) {
          val offset = event.offset + event.newFragment.length
          completionScheduler.schedule(editor, offset)
        }
      }
    })

    editor.project?.messageBus?.connect()?.let {
      it.subscribe(
        FileEditorManagerListener.FILE_EDITOR_MANAGER,
        object: FileEditorManagerListener {
          override fun selectionChanged(event: FileEditorManagerEvent) {
            logger.info("FileEditorManagerListener selectionChanged.")
            completionScheduler.clear()
          }
        }
      )
      messagesConnection[editor] = it
    }
  }

  override fun editorReleased(event: EditorFactoryEvent) {
    messagesConnection[event.editor]?.let {
      it.disconnect()
      it.dispose()
    }
  }
}