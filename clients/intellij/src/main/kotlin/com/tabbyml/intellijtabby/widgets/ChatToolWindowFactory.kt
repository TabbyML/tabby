package com.tabbyml.intellijtabby.widgets

import com.intellij.openapi.components.service
import com.intellij.openapi.diagnostic.logger
import com.intellij.openapi.project.DumbAware
import com.intellij.openapi.project.Project
import com.intellij.openapi.wm.ToolWindow
import com.intellij.openapi.wm.ToolWindowFactory
import com.intellij.ui.components.JBLabel
import com.intellij.ui.content.ContentFactory
import com.intellij.util.ui.JBUI
import com.tabbyml.intellijtabby.chat.ChatBrowserFactory
import java.awt.GridBagLayout
import javax.swing.JPanel

class ChatToolWindowFactory : ToolWindowFactory, DumbAware {
  private val logger = logger<ChatToolWindowFactory>()

  override fun createToolWindowContent(project: Project, toolWindow: ToolWindow) {
    try {
      val chatBrowserFactory = project.service<ChatBrowserFactory>()
      val browser = chatBrowserFactory.createChatBrowser(toolWindow)
      val content = ContentFactory.getInstance().createContent(browser.component, "", false)
      toolWindow.contentManager.addContent(content)
    } catch (e: Exception) {
      logger.warn("Failed to create chat tool window", e)

      val helpMessage =
        """
          <html>
          Failed to create the chat panel.<br/>
          Please check the <a href="https://tabby.tabbyml.com/docs/extensions/troubleshooting/#check-browser-compatibility-in-intellij-platform-ides">online documentation</a> for trouble shooting.
          </html>
        """.trimIndent()
      val label = JBLabel(helpMessage).apply {
        setBorder(JBUI.Borders.emptyLeft(20))
        setCopyable(true)
      }
      val panel = JPanel().apply {
        setLayout(GridBagLayout())
        add(label)
      }

      val content = ContentFactory.getInstance().createContent(panel, "", false)
      toolWindow.contentManager.addContent(content)
    }
  }

  companion object {
    const val TOOL_WINDOW_ID = "Tabby"
  }
}