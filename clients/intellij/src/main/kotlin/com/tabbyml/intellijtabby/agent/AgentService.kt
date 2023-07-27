package com.tabbyml.intellijtabby.agent

import com.intellij.openapi.application.ReadAction
import com.intellij.openapi.components.Service
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.editor.Editor
import com.intellij.psi.PsiDocumentManager
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

  fun getCompletion(editor: Editor, offset: Int): CompletableFuture<Agent.CompletionResponse>? {
    return agent.thenCompose {agent ->
      ReadAction.compute<PsiFile, Throwable> {
        editor.project?.let { project ->
          PsiDocumentManager.getInstance(project).getPsiFile(editor.document)
        }
      }?.let { file ->
        agent?.getCompletions(Agent.CompletionRequest(
          file.virtualFile.path,
          file.language.id, // FIXME: map language id
          editor.document.text,
          offset
        ))
      }
    }
  }
}