package com.tabbyml.intellijtabby.notifications

import com.intellij.ide.BrowserUtil
import com.intellij.notification.Notification
import com.intellij.notification.NotificationType
import com.intellij.notification.Notifications
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.application.invokeLater
import com.intellij.openapi.options.ShowSettingsUtil
import com.tabbyml.intellijtabby.lsp.ConnectionService
import com.tabbyml.intellijtabby.settings.Configurable

fun notifyInitializationFailed(exception: ConnectionService.InitializationException) {
  val notification = Notification(
    "com.tabbyml.intellijtabby.notifications.warning",
    "Tabby initialization failed",
    "${exception.message}",
    NotificationType.ERROR,
  )
  notification.addAction(object : AnAction("Open Online Documentation") {
    override fun actionPerformed(e: AnActionEvent) {
      notification.expire()
      BrowserUtil.browse("https://tabby.tabbyml.com/docs/extensions/troubleshooting/#tabby-initialization-failed")
    }
  })
  invokeLater {
    Notifications.Bus.notify(notification)
  }
}

fun notifyAuthRequired() {
  val notification = Notification(
    "com.tabbyml.intellijtabby.notifications.warning",
    "Tabby server requires authentication, please set your personal token.",
    NotificationType.WARNING,
  )
  notification.addAction(object : AnAction("Open Settings...") {
    override fun actionPerformed(e: AnActionEvent) {
      notification.expire()
      ShowSettingsUtil.getInstance().showSettingsDialog(e.project, Configurable::class.java)
    }
  })
  invokeLater {
    Notifications.Bus.notify(notification)
  }
}