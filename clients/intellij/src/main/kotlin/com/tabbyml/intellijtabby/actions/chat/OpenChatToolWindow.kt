package com.tabbyml.intellijtabby.actions.chat

import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.tabbyml.intellijtabby.widgets.openChatToolWindow

class OpenChatToolWindow : AnAction() {
  override fun actionPerformed(e: AnActionEvent) {
    e.project?.let { openChatToolWindow(it) }
  }
}