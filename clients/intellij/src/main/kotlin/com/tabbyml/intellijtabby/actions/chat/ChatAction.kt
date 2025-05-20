package com.tabbyml.intellijtabby.actions.chat

import com.intellij.openapi.actionSystem.DataContext
import com.intellij.openapi.components.serviceOrNull
import com.intellij.openapi.editor.Caret
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.actionSystem.EditorAction
import com.intellij.openapi.editor.actionSystem.EditorActionHandler
import com.intellij.openapi.project.Project
import com.tabbyml.intellijtabby.chat.ChatBrowserFactory
import com.tabbyml.intellijtabby.events.FeaturesState
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
      return isChatFeatureEnabled(editor.project) && chatActionHandler.isEnabled(editor, chatBrowser)
    }
  })

fun isChatFeatureEnabled(project: Project?): Boolean {
  if (project == null) return false
  val featuresState = project.serviceOrNull<FeaturesState>()
  return featuresState?.features?.chat ?: false
}