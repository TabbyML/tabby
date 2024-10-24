package com.tabbyml.intellijtabby.widgets

import com.intellij.openapi.components.service
import com.intellij.openapi.project.DumbAware
import com.intellij.openapi.project.Project
import com.intellij.openapi.wm.ToolWindow
import com.intellij.openapi.wm.ToolWindowFactory
import com.intellij.ui.content.ContentFactory
import com.tabbyml.intellijtabby.chat.ChatBrowserFactory

class ChatToolWindowFactory : ToolWindowFactory, DumbAware {
  override fun createToolWindowContent(project: Project, toolWindow: ToolWindow) {
    val chatBrowserFactory = project.service<ChatBrowserFactory>()
    val browser = chatBrowserFactory.createChatBrowser(toolWindow)
    val content = ContentFactory.getInstance().createContent(browser.component, "", false)
    toolWindow.contentManager.addContent(content)
  }

  companion object {
    const val TOOL_WINDOW_ID = "Tabby"
  }
}