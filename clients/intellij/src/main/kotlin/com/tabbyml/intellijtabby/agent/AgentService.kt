package com.tabbyml.intellijtabby.agent

import com.intellij.openapi.application.ReadAction
import com.intellij.openapi.components.Service
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.editor.Editor
import com.intellij.psi.PsiDocumentManager
import com.intellij.psi.PsiFile
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

@Service
class AgentService {
  private val logger = Logger.getInstance(AgentService::class.java)
  private var agent: Agent = Agent()
  val scope: CoroutineScope = CoroutineScope(Dispatchers.IO)

  init {
    scope.launch {
      try {
        agent.initialize()
        logger.info("Agent init done.")
      } catch (e: Error) {
        logger.error("Agent init failed: $e")
      }
    }
  }

  private suspend fun waitForInitialized() {
    agent.status.first { it != Agent.Status.NOT_INITIALIZED }
  }

  suspend fun getCompletion(editor: Editor, offset: Int): Agent.CompletionResponse? {
    waitForInitialized()
    return ReadAction.compute<PsiFile, Throwable> {
      editor.project?.let { project ->
        PsiDocumentManager.getInstance(project).getPsiFile(editor.document)
      }
    }?.let { file ->
      agent.getCompletions(
        Agent.CompletionRequest(
          file.virtualFile.path,
          file.language.id, // FIXME: map language id
          editor.document.text,
          offset
        )
      )
    }
  }

  suspend fun postEvent() {
    waitForInitialized()
  }

  // FIXME: dispose agent
  fun dispose() {
    agent.close()
  }
}