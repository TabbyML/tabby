package com.tabbyml.intellijtabby.actions.chat

import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.components.service
import com.intellij.openapi.fileEditor.FileEditorManager
import com.intellij.openapi.wm.IdeFocusManager
import com.intellij.openapi.wm.ToolWindowManager
import com.tabbyml.intellijtabby.actionPromoter.HasPriority
import com.tabbyml.intellijtabby.chat.ChatBrowser
import com.tabbyml.intellijtabby.chat.ChatBrowserFactory
import com.tabbyml.intellijtabby.widgets.ChatToolWindowFactory

class ToggleChatToolWindow : AnAction(), HasPriority {
  override fun actionPerformed(e: AnActionEvent) {
    val project = e.project ?: return
    val toolWindowManager = ToolWindowManager.getInstance(project)
    val toolWindow = toolWindowManager.getToolWindow(ChatToolWindowFactory.TOOL_WINDOW_ID) ?: return

    val editor = FileEditorManager.getInstance(project).selectedTextEditor
    val chatBrowserFactory = project.service<ChatBrowserFactory>()
    val chatBrowser = chatBrowserFactory.getChatBrowser(toolWindow)
    if (toolWindow.isActive) {
      if (editor != null) {
        IdeFocusManager.getInstance(project).requestFocus(editor.contentComponent, true)
      }
    } else {
      toolWindow.show {
        toolWindow.activate {
          if (editor != null && chatBrowser != null && editor.selectionModel.let { it.hasSelection() && !it.selectedText.isNullOrBlank() }) {
            chatBrowser.addActiveEditorAsContext(ChatBrowser.RangeStrategy.SELECTION)
          }
        }
      }
    }
  }

  override val priority: Int = 1
}