package com.tabbyml.intellijtabby.actions.chat

import com.intellij.openapi.editor.Editor
import com.tabbyml.intellijtabby.chat.ChatBrowser

interface ChatActionHandler {
  fun doExecute(editor: Editor, chatBrowser: ChatBrowser)
  fun isEnabled(editor: Editor, chatBrowser: ChatBrowser?): Boolean {
    return true
  }
}