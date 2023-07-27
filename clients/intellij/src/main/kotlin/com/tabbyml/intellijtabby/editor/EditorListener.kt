package com.tabbyml.intellijtabby.editor

import com.intellij.openapi.components.service
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.editor.event.*
import com.intellij.openapi.fileEditor.FileEditorManager

class EditorListener : EditorFactoryListener {
  private val logger = Logger.getInstance(EditorListener::class.java)

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
  }
}