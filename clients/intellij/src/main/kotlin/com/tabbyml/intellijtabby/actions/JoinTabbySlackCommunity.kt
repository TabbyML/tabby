package com.tabbyml.intellijtabby.actions

import com.intellij.ide.BrowserUtil
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent

class JoinTabbySlackCommunity : AnAction() {
  override fun actionPerformed(e: AnActionEvent) {
    BrowserUtil.browse("https://links.tabbyml.com/join-slack-extensions")
  }
}