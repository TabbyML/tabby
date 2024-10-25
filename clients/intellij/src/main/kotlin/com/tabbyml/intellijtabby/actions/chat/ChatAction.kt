package com.tabbyml.intellijtabby.actions.chat

import com.intellij.openapi.actionSystem.DataContext
import com.intellij.openapi.components.service
import com.intellij.openapi.editor.Caret
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.actionSystem.EditorAction
import com.intellij.openapi.editor.actionSystem.EditorActionHandler
import com.intellij.openapi.project.Project
import com.intellij.openapi.wm.ToolWindowManager
import com.tabbyml.intellijtabby.chat.ChatBrowser
import com.tabbyml.intellijtabby.chat.ChatBrowserFactory
import com.tabbyml.intellijtabby.widgets.ChatToolWindowFactory

abstract class ChatAction(private val chatActionHandler: ChatActionHandler) :
  EditorAction(object : EditorActionHandler() {
    private fun openChatToolWindow(project: Project, runnable: Runnable?) {
      val toolWindowManager = ToolWindowManager.getInstance(project)
      val toolWindow = toolWindowManager.getToolWindow(ChatToolWindowFactory.TOOL_WINDOW_ID) ?: return
      toolWindow.show(runnable)
    }

    private fun findActiveChatBrowser(editor: Editor): ChatBrowser? {
      val project = editor.project ?: return null
      val toolWindowManager = ToolWindowManager.getInstance(project)
      val toolWindow = toolWindowManager.getToolWindow(ChatToolWindowFactory.TOOL_WINDOW_ID) ?: return null
      val chatBrowserFactory = project.service<ChatBrowserFactory>()
      return chatBrowserFactory.getChatBrowser(toolWindow)
    }

    override fun doExecute(editor: Editor, caret: Caret?, dataContext: DataContext?) {
      val project = editor.project ?: return
      openChatToolWindow(project) {
        val chatBrowser = findActiveChatBrowser(editor) ?: return@openChatToolWindow
        chatActionHandler.doExecute(editor, chatBrowser)
      }
    }

    override fun isEnabledForCaret(editor: Editor, caret: Caret, dataContext: DataContext?): Boolean {
      val chatBrowser = findActiveChatBrowser(editor)
      return chatActionHandler.isEnabled(editor, chatBrowser)
    }
  })