package com.tabbyml.intellijtabby.actions

import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.wm.ToolWindow
import com.intellij.openapi.wm.ToolWindowManager

class OpenChatToolWindow : AnAction() {
  override fun actionPerformed(e: AnActionEvent) {
    val toolWindowManager: ToolWindowManager = e.project?.let { ToolWindowManager.getInstance(it) } ?: return
    val toolWindow: ToolWindow = toolWindowManager.getToolWindow("Tabby") ?: return
    toolWindow.show()
  }
}