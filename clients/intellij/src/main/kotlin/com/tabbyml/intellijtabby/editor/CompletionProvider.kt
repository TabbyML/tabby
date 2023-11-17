package com.tabbyml.intellijtabby.editor

import com.intellij.openapi.components.Service
import com.intellij.openapi.components.service
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.editor.Editor
import com.tabbyml.intellijtabby.agent.AgentService
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

@Service
class CompletionProvider {
  private val logger = Logger.getInstance(CompletionProvider::class.java)

  data class CompletionContext(val editor: Editor, val offset: Int, val job: Job)

  private val ongoingCompletionFlow: MutableStateFlow<CompletionContext?> = MutableStateFlow(null)
  val ongoingCompletion = ongoingCompletionFlow.asStateFlow()

  fun provideCompletion(editor: Editor, offset: Int, manually: Boolean = false) {
    val agentService = service<AgentService>()
    val inlineCompletionService = service<InlineCompletionService>()
    clear()
    val job = agentService.scope.launch {
      logger.info("Trigger completion at $offset")
      agentService.provideCompletion(editor, offset, manually).let {
        ongoingCompletionFlow.value = null
        if (it != null) {
          logger.info("Show completion at $offset: $it")
          inlineCompletionService.show(editor, offset, it)
        }
      }
    }
    ongoingCompletionFlow.value = CompletionContext(editor, offset, job)
  }

  fun clear() {
    val inlineCompletionService = service<InlineCompletionService>()
    inlineCompletionService.dismiss()
    ongoingCompletionFlow.value?.let {
      it.job.cancel()
      ongoingCompletionFlow.value = null
    }
  }
}