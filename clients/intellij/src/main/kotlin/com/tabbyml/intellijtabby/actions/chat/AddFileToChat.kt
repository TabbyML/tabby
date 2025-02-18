package com.tabbyml.intellijtabby.actions.chat

import com.intellij.openapi.editor.Editor
import com.tabbyml.intellijtabby.chat.ChatBrowser

class AddFileToChat : ChatAction(object : ChatActionHandler {
  override fun doExecute(editor: Editor, chatBrowser: ChatBrowser) {
    chatBrowser.addActiveEditorAsContext(ChatBrowser.RangeStrategy.FILE)
  }
})