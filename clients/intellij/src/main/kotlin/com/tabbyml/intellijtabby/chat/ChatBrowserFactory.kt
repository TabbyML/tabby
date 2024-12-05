package com.tabbyml.intellijtabby.chat

import com.intellij.openapi.Disposable
import com.intellij.openapi.components.Service
import com.intellij.openapi.project.Project
import com.intellij.openapi.wm.ToolWindow

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
}