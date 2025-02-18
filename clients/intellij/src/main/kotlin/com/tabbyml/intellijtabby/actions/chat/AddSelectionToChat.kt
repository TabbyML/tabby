package com.tabbyml.intellijtabby.actions.chat

import com.intellij.openapi.editor.Editor
import com.tabbyml.intellijtabby.chat.ChatBrowser

class AddSelectionToChat : ChatAction(object : ChatActionHandler {
  override fun doExecute(editor: Editor, chatBrowser: ChatBrowser) {
    chatBrowser.addActiveEditorAsContext(ChatBrowser.RangeStrategy.SELECTION)
  }

  override fun isEnabled(editor: Editor, chatBrowser: ChatBrowser?): Boolean {
    return editor.selectionModel.let { it.hasSelection() && !it.selectedText.isNullOrBlank() }
  }
})