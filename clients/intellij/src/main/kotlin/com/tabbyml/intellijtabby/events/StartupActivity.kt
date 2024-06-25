package com.tabbyml.intellijtabby.events

import com.intellij.openapi.components.service
import com.intellij.openapi.project.Project
import com.intellij.openapi.startup.ProjectActivity
import com.tabbyml.intellijtabby.completion.InlineCompletionService
import com.tabbyml.intellijtabby.lsp.ConnectionService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class StartupActivity : ProjectActivity {
  override suspend fun execute(project: Project) {
    // initialize services
    val connectionService = project.service<ConnectionService>()
    CoroutineScope(Dispatchers.IO).launch {
      connectionService.getServerAsync()
    }
    project.service<CombinedState>()
    project.service<InlineCompletionService>()
  }
}