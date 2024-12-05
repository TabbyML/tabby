package com.tabbyml.intellijtabby.actions.chat

import com.intellij.openapi.editor.Editor
import com.tabbyml.intellijtabby.chat.ChatBrowser

class Fix : ChatAction(object : ChatActionHandler {
  override fun doExecute(editor: Editor, chatBrowser: ChatBrowser) {
    chatBrowser.fixSelectedText()
  }

  override fun isEnabled(editor: Editor, chatBrowser: ChatBrowser?): Boolean {
    return editor.selectionModel.let { it.hasSelection() && !it.selectedText.isNullOrBlank() }
  }
})