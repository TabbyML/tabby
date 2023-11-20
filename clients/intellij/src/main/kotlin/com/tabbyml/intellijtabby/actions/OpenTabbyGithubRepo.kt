package com.tabbyml.intellijtabby.actions

import com.intellij.ide.BrowserUtil
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent

class OpenTabbyGithubRepo : AnAction() {
  override fun actionPerformed(e: AnActionEvent) {
    BrowserUtil.browse("https://github.com/tabbyml/tabby")
  }
}