package com.tabbyml.intellijtabby.notifications

import com.intellij.ide.BrowserUtil
import com.intellij.notification.Notification
import com.intellij.notification.NotificationType
import com.intellij.notification.Notifications
import com.intellij.openapi.actionSystem.ActionGroup
import com.intellij.openapi.actionSystem.ActionManager
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.application.invokeLater
import com.intellij.openapi.options.ShowSettingsUtil
import com.intellij.openapi.ui.popup.JBPopupFactory
import com.tabbyml.intellijtabby.lsp.ConnectionService
import com.tabbyml.intellijtabby.settings.Configurable

var initializationFailedNotification: Notification? = null

fun notifyInitializationFailed(exception: ConnectionService.InitializationException) {
  initializationFailedNotification?.expire()

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
  initializationFailedNotification = notification
  invokeLater {
    Notifications.Bus.notify(notification)
  }
}

var authRequiredNotification: Notification? = null

fun notifyAuthRequired() {
  authRequiredNotification?.expire()
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
  notification.addAction(object : AnAction("Open Online Help") {
    override fun actionPerformed(e: AnActionEvent) {
      notification.expire()
      BrowserUtil.browse("https://tabby.tabbyml.com/docs/quick-start/register-account/")
    }
  })
  authRequiredNotification = notification
  invokeLater {
    Notifications.Bus.notify(notification)
  }
}

fun hideAuthRequiredNotification() {
  authRequiredNotification?.expire()
}

fun showOnlineHelp(e: AnActionEvent) {
  e.project?.let {
    invokeLater {
      val actionManager = ActionManager.getInstance()
      val actionGroup = actionManager.getAction("Tabby.OpenOnlineHelp") as ActionGroup
      val popup = JBPopupFactory.getInstance().createActionGroupPopup(
        "Online Help",
        actionGroup,
        e.dataContext,
        false,
        null,
        10,
      )
      popup.showCenteredInCurrentWindow(it)
    }
  }
}
