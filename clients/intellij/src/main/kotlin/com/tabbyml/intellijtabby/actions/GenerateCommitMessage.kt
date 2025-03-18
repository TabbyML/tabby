package com.tabbyml.intellijtabby.actions

import com.intellij.openapi.actionSystem.ActionUpdateThread
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.actionSystem.CommonDataKeys
import com.intellij.openapi.application.invokeLater
import com.intellij.openapi.components.serviceOrNull
import com.intellij.openapi.progress.ProgressIndicator
import com.intellij.openapi.progress.ProgressManager
import com.intellij.openapi.progress.Task
import com.intellij.openapi.project.guessProjectDir
import com.intellij.openapi.ui.Messages
import com.intellij.openapi.wm.WindowManager
import com.tabbyml.intellijtabby.events.FeaturesState
import com.tabbyml.intellijtabby.git.GitProvider
import com.tabbyml.intellijtabby.lsp.ConnectionService
import com.tabbyml.intellijtabby.lsp.protocol.GenerateCommitMessageParams
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.future.await
import kotlinx.coroutines.launch
import java.awt.BorderLayout
import java.awt.Toolkit
import java.awt.datatransfer.StringSelection
import javax.swing.JOptionPane
import javax.swing.JPanel
import javax.swing.JScrollPane
import javax.swing.JTextArea

class GenerateCommitMessage : AnAction() {
  private val scope = CoroutineScope(Dispatchers.IO)

  override fun actionPerformed(e: AnActionEvent) {
    val project = e.getRequiredData(CommonDataKeys.PROJECT)
    val projectDir = project.guessProjectDir()?.url ?: return

    val task = object : Task.Modal(
      project, null, "Generate Commit Message", true
    ) {
      lateinit var job: Job
      override fun run(indicator: ProgressIndicator) {
        job = scope.launch {
          indicator.isIndeterminate = true
          indicator.text = "Generating commit message..."

          val parentComponent = WindowManager.getInstance().getFrame(project)
          val server = project.serviceOrNull<ConnectionService>()?.getServerAsync() ?: return@launch

          val commitMessage = try {
            val result = server.chatFeature.generateCommitMessage(GenerateCommitMessageParams(projectDir)).await()
            if (result?.commitMessage.isNullOrBlank()) {
              throw NoCommitMessageGeneratedException("No commit message generated.")
            } else {
              result.commitMessage
            }
          } catch (e: Exception) {
            invokeLater {
              Messages.showErrorDialog(
                parentComponent,
                if (e is NoCommitMessageGeneratedException) {
                  e.message
                } else {
                  "Failed to generate commit message. ${e.message}"
                },
                "Generate Commit Message"
              )
            }
            return@launch
          }

          invokeLater {
            val textArea = JTextArea(commitMessage, 10, 80)
            textArea.lineWrap = true
            val panel = JPanel(BorderLayout())
            panel.add(JScrollPane(textArea), BorderLayout.CENTER)

            val selection = JOptionPane.showOptionDialog(
              parentComponent,
              panel,
              "Generate Commit Message",
              JOptionPane.OK_CANCEL_OPTION,
              JOptionPane.PLAIN_MESSAGE,
              null,
              arrayOf("Copy", "Cancel"),
              "Copy"
            )

            if (selection == 0) {
              val stringSelection = StringSelection(textArea.text)
              val clipboard = Toolkit.getDefaultToolkit().systemClipboard
              clipboard.setContents(stringSelection, null)
            }
          }
        }

        while (job.isActive) {
          indicator.checkCanceled()
          Thread.sleep(100)
        }
      }

      override fun onCancel() {
        job.cancel()
      }
    }

    ProgressManager.getInstance().run(task)
  }

  class NoCommitMessageGeneratedException(message: String) : Exception(message)

  override fun update(e: AnActionEvent) {
    val project = e.project ?: e.getData(CommonDataKeys.PROJECT)
    val projectDir = project?.guessProjectDir()?.url

    val gitProvider = project?.serviceOrNull<GitProvider>()
    val isSupported = gitProvider?.isSupported()
    e.presentation.isVisible = (isSupported ?: false) && (projectDir != null)

    val featuresState = project?.serviceOrNull<FeaturesState>()
    e.presentation.isEnabled = featuresState?.features?.chat ?: false
  }

  override fun getActionUpdateThread(): ActionUpdateThread {
    return ActionUpdateThread.BGT
  }
}