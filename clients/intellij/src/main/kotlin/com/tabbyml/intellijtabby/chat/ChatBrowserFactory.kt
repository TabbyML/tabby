package com.tabbyml.intellijtabby.chat

import com.intellij.openapi.Disposable
import com.intellij.openapi.components.Service
import com.intellij.openapi.components.service
import com.intellij.openapi.project.Project
import com.intellij.openapi.wm.ToolWindow
import com.intellij.openapi.wm.ToolWindowManager
import com.tabbyml.intellijtabby.widgets.ChatToolWindowFactory

@Service(Service.Level.PROJECT)
class ChatBrowserFactory(private val project: Project) : Disposable {
  private val registry: MutableMap<ToolWindow, ChatBrowser> = mutableMapOf()

  fun createChatBrowser(toolWindow: ToolWindow): ChatBrowser {
    val chatBrowser = ChatBrowser(project)
    registry[toolWindow] = chatBrowser
    return chatBrowser
  }

  fun getChatBrowser(toolWindow: ToolWindow): ChatBrowser? {
    return registry[toolWindow]
  }

  override fun dispose() {
    registry.forEach {
      it.value.dispose()
    }
    registry.clear()
  }

  companion object {
    fun findActiveChatBrowser(project: Project): ChatBrowser? {
      val toolWindowManager = ToolWindowManager.getInstance(project)
      val toolWindow = toolWindowManager.getToolWindow(ChatToolWindowFactory.TOOL_WINDOW_ID) ?: return null
      val chatBrowserFactory = project.service<ChatBrowserFactory>()
      return chatBrowserFactory.getChatBrowser(toolWindow)
    }
  }
}