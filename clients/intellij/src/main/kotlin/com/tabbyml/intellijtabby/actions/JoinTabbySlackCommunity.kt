package com.tabbyml.intellijtabby.actions

import com.intellij.ide.BrowserUtil
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent

class JoinTabbySlackCommunity: AnAction() {
  override fun actionPerformed(e: AnActionEvent) {
    BrowserUtil.browse("https://join.slack.com/t/tabbycommunity/shared_invite/zt-1xeiddizp-bciR2RtFTaJ37RBxr8VxpA")
  }
}