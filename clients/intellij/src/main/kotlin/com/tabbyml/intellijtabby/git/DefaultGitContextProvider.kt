package com.tabbyml.intellijtabby.git

import com.intellij.openapi.project.Project
import com.intellij.openapi.vfs.VirtualFile
import com.tabbyml.intellijtabby.agent.Agent

class DefaultGitContextProvider : GitContextProvider {
    override fun getGitContextForFile(project: Project, file: VirtualFile): Agent.CompletionRequest.GitContext? {
        return null
    }
}