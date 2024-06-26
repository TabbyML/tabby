package com.tabbyml.intellijtabby.events

import com.intellij.openapi.editor.Document
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.event.DocumentEvent
import com.intellij.util.messages.Topic

interface DocumentListener {
  fun documentOpened(document: Document, editor: Editor) {}
  fun documentClosed(document: Document, editor: Editor) {}
  fun documentChanged(document: Document, editor: Editor, event: DocumentEvent) {}

  companion object {
    @Topic.ProjectLevel
    val TOPIC = Topic(DocumentListener::class.java, Topic.BroadcastDirection.NONE)
  }
}
