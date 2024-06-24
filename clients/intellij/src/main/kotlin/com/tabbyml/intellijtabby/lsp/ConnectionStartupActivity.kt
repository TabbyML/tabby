package com.tabbyml.intellijtabby.lsp

import com.intellij.openapi.components.service
import com.intellij.openapi.project.Project
import com.intellij.openapi.startup.ProjectActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class ConnectionStartupActivity : ProjectActivity {
  override suspend fun execute(project: Project) {
    val connectionService = project.service<ConnectionService>()
    CoroutineScope(Dispatchers.IO).launch {
      connectionService.getServerAsync()
    }
  }
}