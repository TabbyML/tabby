package com.tabbyml.intellijtabby.events

import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.event.CaretEvent
import com.intellij.util.messages.Topic

interface CaretListener {
  fun caretPositionChanged(editor: Editor, event: CaretEvent) {}

  companion object {
    @Topic.ProjectLevel
    val TOPIC = Topic(CaretListener::class.java, Topic.BroadcastDirection.NONE)
  }
}