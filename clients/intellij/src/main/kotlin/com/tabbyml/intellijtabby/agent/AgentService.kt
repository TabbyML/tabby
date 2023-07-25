package com.tabbyml.intellijtabby.agent

import com.intellij.openapi.components.Service
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.editor.Editor
import com.intellij.psi.PsiFile
import java.util.concurrent.CompletableFuture

@Service
class AgentService {
  private val logger = Logger.getInstance(AgentService::class.java)
  private val agent: CompletableFuture<Agent?> = CompletableFuture<Agent?>()

  init {
    try {
      val instance = Agent()
      instance.initialize().thenApply {
        logger.info("Agent init done: $it")
        agent.complete(instance)
      }
    } catch (_: Error) {
      agent.complete(null)
    }
  }

  fun getCompletion(editor: Editor, file: PsiFile, offset: Int): CompletableFuture<Agent.CompletionResponse>? {
    return agent.thenCompose {
      it?.getCompletions(Agent.CompletionRequest(
        file.virtualFile.path,
        file.language.id, // FIXME: map language id
        editor.document.text,
        offset
      ))
    }
  }
}