package com.tabbyml.intellijtabby.agent

import com.intellij.lang.Language
import com.intellij.openapi.Disposable
import com.intellij.openapi.application.ReadAction
import com.intellij.openapi.components.PersistentStateComponent
import com.intellij.openapi.components.Service
import com.intellij.openapi.components.State
import com.intellij.openapi.components.service
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.diagnostic.logger
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.fileTypes.impl.AbstractFileType
import com.intellij.psi.PsiDocumentManager
import com.intellij.psi.PsiFile
import com.tabbyml.intellijtabby.agent.AgentService.Companion.getLanguageId
import com.tabbyml.intellijtabby.settings.ApplicationSettingsState
import io.ktor.util.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

@Service
class AgentService : Disposable {
  private val logger = Logger.getInstance(AgentService::class.java)
  private var agent: Agent = Agent()
  val scope: CoroutineScope = CoroutineScope(Dispatchers.IO)

  init {
    scope.launch {
      try {
        agent.initialize(createAgentConfig())
        logger.info("Agent init done.")
      } catch (e: Error) {
        logger.error("Agent init failed: $e")
      }
    }
  }

  private fun createAgentConfig(): Agent.Config {
    val appSettings = service<ApplicationSettingsState>()
    return Agent.Config(
      server = if (appSettings.serverEndpoint.isNotBlank()) {
        Agent.Config.Server(
          endpoint = appSettings.serverEndpoint,
        )
      } else {
        null
      },
      anonymousUsageTracking = if (appSettings.isAnonymousUsageTrackingDisabled) {
        Agent.Config.AnonymousUsageTracking(
          disabled = true,
        )
      } else {
        null
      },
    )
  }

  private suspend fun waitForInitialized() {
    agent.status.first { it != Agent.Status.NOT_INITIALIZED }
  }

  suspend fun updateConfig() {
    waitForInitialized()
    agent.updateConfig(createAgentConfig())
  }

  suspend fun getCompletion(editor: Editor, offset: Int): Agent.CompletionResponse? {
    waitForInitialized()
    return ReadAction.compute<PsiFile, Throwable> {
      editor.project?.let { project ->
        PsiDocumentManager.getInstance(project).getPsiFile(editor.document)
      }
    }?.let { file ->
      logger.info("Language id: ${file.language}")
      agent.getCompletions(
        Agent.CompletionRequest(
          file.virtualFile.path,
          file.getLanguageId(),
          editor.document.text,
          offset
        )
      )
    }
  }

  suspend fun postEvent() {
    waitForInitialized()
  }

  override fun dispose() {
    agent.close()
  }

  companion object {
    // Language id: https://code.visualstudio.com/docs/languages/identifiers
    private fun PsiFile.getLanguageId(): String {
      if (this.language != Language.ANY
        && this.language.id.toLowerCasePreservingASCIIRules() !in arrayOf("txt", "text", "textmate")
      ) {
        if (languageIdMap.containsKey(this.language.id)) {
          return languageIdMap[this.language.id]!!
        }
        return this.language.id.toLowerCasePreservingASCIIRules()
          .replace("#", "sharp")
          .replace("++", "pp")
          .replace(" ", "")
      }
      return if (filetypeMap.containsKey(this.fileType.defaultExtension)) {
        filetypeMap[this.fileType.defaultExtension]!!
      } else {
        this.fileType.defaultExtension.toLowerCasePreservingASCIIRules()
      }
    }

    private val languageIdMap = mapOf(
      "ObjectiveC" to "objective-c",
      "ObjectiveC++" to "objective-cpp",
    )
    private val filetypeMap = mapOf(
      "py" to "python",
      "js" to "javascript",
      "cjs" to "javascript",
      "mjs" to "javascript",
      "jsx" to "javascriptreact",
      "ts" to "typescript",
      "tsx" to "typescriptreact",
      "kt" to "kotlin",
      "md" to "markdown",
      "cc" to "cpp",
      "cs" to "csharp",
      "m" to "objective-c",
      "mm" to "objective-cpp",
      "sh" to "shellscript",
      "zsh" to "shellscript",
      "bash" to "shellscript",
      "txt" to "plaintext",
    )
  }
}