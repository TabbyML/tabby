package com.tabbyml.intellijtabby.git

import com.intellij.openapi.project.Project
import com.intellij.openapi.vfs.VirtualFile
import com.tabbyml.intellijtabby.agent.Agent

interface GitContextProvider {
    fun getGitContextForFile(project: Project, file: VirtualFile): Agent.CompletionRequest.GitContext?
}