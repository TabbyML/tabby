package com.tabbyml.intellijtabby.actions.chat

import com.intellij.openapi.actionSystem.ActionUpdateThread
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.tabbyml.intellijtabby.chat.ChatBrowserFactory

abstract class ChatNavigationAction(private val view: String) : AnAction() {
  override fun actionPerformed(e: AnActionEvent) {
    val chatBrowser = e.project?.let { ChatBrowserFactory.findActiveChatBrowser(it) }
    chatBrowser?.navigate(view)
  }

  override fun update(e: AnActionEvent) {
    val chatBrowser = e.project?.let { ChatBrowserFactory.findActiveChatBrowser(it) }
    e.presentation.isEnabled = chatBrowser?.isChatPanelLoaded == true
  }

  override fun getActionUpdateThread(): ActionUpdateThread {
    return ActionUpdateThread.BGT
  }
}