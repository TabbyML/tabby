package com.tabbyml.intellijtabby.actions.chat

import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.wm.ToolWindowManager
import com.tabbyml.intellijtabby.widgets.ChatToolWindowFactory

class OpenChatToolWindow : AnAction() {
  override fun actionPerformed(e: AnActionEvent) {
    val toolWindowManager = e.project?.let { ToolWindowManager.getInstance(it) } ?: return
    val toolWindow = toolWindowManager.getToolWindow(ChatToolWindowFactory.TOOL_WINDOW_ID) ?: return
    toolWindow.show()
  }
}