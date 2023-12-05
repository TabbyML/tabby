package com.tabbyml.intellijtabby.editor

import com.intellij.openapi.application.invokeLater
import com.intellij.openapi.components.service
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.event.*
import com.intellij.openapi.fileEditor.FileEditorManager
import com.intellij.openapi.fileEditor.FileEditorManagerEvent
import com.intellij.openapi.fileEditor.FileEditorManagerListener
import com.intellij.util.messages.MessageBusConnection
import com.tabbyml.intellijtabby.settings.ApplicationSettingsState

class EditorListener : EditorFactoryListener {
  private val logger = Logger.getInstance(EditorListener::class.java)
  private val messagesConnection = mutableMapOf<Editor, MessageBusConnection>()

  override fun editorCreated(event: EditorFactoryEvent) {
    val editor = event.editor
    val editorManager = editor.project?.let { FileEditorManager.getInstance(it) } ?: return
    val settings = service<ApplicationSettingsState>()
    val completionProvider = service<CompletionProvider>()

    editor.caretModel.addCaretListener(object : CaretListener {
      override fun caretPositionChanged(event: CaretEvent) {
        logger.debug("CaretListener: caretPositionChanged $event")
        if (editorManager.selectedTextEditor == editor) {
          completionProvider.ongoingCompletion.value.let {
            if (it != null && it.editor == editor && it.offset == editor.caretModel.primaryCaret.offset) {
              // keep ongoing completion
              logger.debug("Keep ongoing completion.")
            } else {
              completionProvider.clear()
            }
          }
        }
      }
    })

    editor.document.addDocumentListener(object : DocumentListener {
      override fun documentChanged(event: DocumentEvent) {
        logger.debug("DocumentListener: documentChanged $event")
        if (editorManager.selectedTextEditor == editor) {
          if (settings.completionTriggerMode == ApplicationSettingsState.TriggerMode.AUTOMATIC) {
            completionProvider.ongoingCompletion.value.let {
              if (it != null && it.editor == editor && it.offset == editor.caretModel.primaryCaret.offset) {
                // keep ongoing completion
                logger.debug("Keep ongoing completion.")
              } else {
                invokeLater {
                  completionProvider.provideCompletion(editor, editor.caretModel.primaryCaret.offset)
                }
              }
            }
          }
        }
      }
    })

    editor.project?.messageBus?.connect()?.let {
      it.subscribe(
        FileEditorManagerListener.FILE_EDITOR_MANAGER,
        object : FileEditorManagerListener {
          override fun selectionChanged(event: FileEditorManagerEvent) {
            logger.debug("FileEditorManagerListener: selectionChanged.")
            completionProvider.clear()
          }
        }
      )
      messagesConnection[editor] = it
    }
  }

  override fun editorReleased(event: EditorFactoryEvent) {
    messagesConnection[event.editor]?.disconnect()
  }
}