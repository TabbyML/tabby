package com.tabbyml.intellijtabby.chat

import com.intellij.openapi.project.DumbAware
import com.intellij.openapi.project.Project
import com.intellij.openapi.wm.ToolWindow
import com.intellij.openapi.wm.ToolWindowFactory
import com.intellij.ui.content.ContentFactory

class TabbyToolWindowFactory : ToolWindowFactory, DumbAware {
  override fun createToolWindowContent(project: Project, toolWindow: ToolWindow) {
    val tabbyBrowser = TabbyBrowser(project)
    val browserComponent = tabbyBrowser.getBrowserComponent()
    val content = ContentFactory.getInstance().createContent(browserComponent, "", false)
    toolWindow.contentManager.addContent(content)
  }

  override fun shouldBeAvailable(project: Project) = true

  init {
    System.setProperty("ide.browser.jcef.contextMenu.devTools.enabled", "true")
  }
}