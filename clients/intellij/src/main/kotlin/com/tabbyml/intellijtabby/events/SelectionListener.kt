package com.tabbyml.intellijtabby.events

import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.event.SelectionEvent
import com.intellij.util.messages.Topic

interface SelectionListener {
  fun selectionChanged(editor: Editor, event: SelectionEvent) {}

  companion object {
    @Topic.ProjectLevel
    val TOPIC = Topic(SelectionListener::class.java, Topic.BroadcastDirection.NONE)
  }
}