package com.tabbyml.intellijtabby.actions.chat

import com.intellij.openapi.actionSystem.DataContext
import com.intellij.openapi.editor.Caret
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.actionSystem.EditorAction
import com.intellij.openapi.editor.actionSystem.EditorActionHandler
import com.tabbyml.intellijtabby.chat.ChatBrowserFactory
import com.tabbyml.intellijtabby.widgets.openChatToolWindow

abstract class ChatAction(private val chatActionHandler: ChatActionHandler) :
  EditorAction(object : EditorActionHandler() {
    override fun doExecute(editor: Editor, caret: Caret?, dataContext: DataContext?) {
      val project = editor.project ?: return
      openChatToolWindow(project) {
        val chatBrowser = ChatBrowserFactory.findActiveChatBrowser(project) ?: return@openChatToolWindow
        chatActionHandler.doExecute(editor, chatBrowser)
      }
    }

    override fun isEnabledForCaret(editor: Editor, caret: Caret, dataContext: DataContext?): Boolean {
      val chatBrowser = editor.project?.let { ChatBrowserFactory.findActiveChatBrowser(it) }
      return chatActionHandler.isEnabled(editor, chatBrowser)
    }
  })