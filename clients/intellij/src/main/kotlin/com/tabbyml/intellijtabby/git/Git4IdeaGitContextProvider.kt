package com.tabbyml.intellijtabby.git

import com.intellij.openapi.project.Project
import com.intellij.openapi.vfs.VirtualFile
import com.tabbyml.intellijtabby.agent.Agent
import git4idea.repo.GitRepositoryManager

class Git4IdeaGitContextProvider : GitContextProvider {
    override fun getGitContextForFile(project: Project, file: VirtualFile): Agent.CompletionRequest.GitContext? {
        val manger = GitRepositoryManager.getInstance(project)
        val repository = manger.getRepositoryForFile(file)
        if (repository === null) return null

        val remotes = ArrayList<Agent.CompletionRequest.GitContext.GitRemote>()

        for (remote in repository.remotes) {
            val url = remote.firstUrl
            if (url !== null) {
                remotes.add(Agent.CompletionRequest.GitContext.GitRemote(remote.name, url))
            }
        }

        if (remotes.isEmpty()) return null

        return Agent.CompletionRequest.GitContext(repository.root.path, remotes)
    }
}