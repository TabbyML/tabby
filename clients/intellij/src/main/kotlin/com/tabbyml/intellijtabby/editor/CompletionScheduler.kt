package com.tabbyml.intellijtabby.editor

import com.intellij.openapi.components.Service
import com.intellij.openapi.components.service
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.fileEditor.FileEditorManagerEvent
import com.intellij.openapi.fileEditor.FileEditorManagerListener
import com.intellij.openapi.project.Project
import com.tabbyml.intellijtabby.agent.AgentService
import java.util.*

@Service
class CompletionScheduler {
  private val logger = Logger.getInstance(CompletionScheduler::class.java)

  data class CompletionContext(val editor: Editor, val offset: Int, val timer: Timer)

  private var project: Project? = null
  var scheduled: CompletionContext? = null
    private set


  fun schedule(editor: Editor, offset: Int) {
    clear()
    val agentService = service<AgentService>()
    val inlineCompletionService = service<InlineCompletionService>()
    inlineCompletionService.dismiss()
    val timer = Timer()
    timer.schedule(object : TimerTask() {
      override fun run() {
        logger.info("Scheduled completion task running")
        agentService.getCompletion(editor, offset)?.thenAccept {
          inlineCompletionService.show(editor, offset, it)
        }
      }
    }, 150)
    scheduled = CompletionContext(editor, offset, timer)

    if (project != editor.project) {
      project = editor.project
      editor.project?.messageBus?.connect()?.subscribe(
        FileEditorManagerListener.FILE_EDITOR_MANAGER,
        object: FileEditorManagerListener {
          override fun selectionChanged(event: FileEditorManagerEvent) {
            logger.info("FileEditorManagerListener selectionChanged.")
            clear()
          }
        }
      )
    }
  }

  fun clear() {
    scheduled?.let {
      it.timer.cancel()
      scheduled = null
    }
    val inlineCompletionService = service<InlineCompletionService>()
    inlineCompletionService.dismiss()
  }
}