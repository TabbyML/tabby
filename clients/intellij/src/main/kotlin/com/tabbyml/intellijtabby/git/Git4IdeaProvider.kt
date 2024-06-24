package com.tabbyml.intellijtabby.git

import com.intellij.openapi.project.Project
import com.intellij.openapi.vfs.VirtualFileManager
import com.tabbyml.intellijtabby.git.GitProvider.Repository
import git4idea.commands.Git
import git4idea.commands.GitCommand
import git4idea.commands.GitLineHandler
import git4idea.repo.GitRepositoryManager


class Git4IdeaProvider(private val project: Project) : GitProvider {
  private val git = Git.getInstance()
  private val virtualFileManager = VirtualFileManager.getInstance()
  private val gitRepositoryManger = GitRepositoryManager.getInstance(project)

  override fun isSupported(): Boolean {
    return true
  }

  override fun getRepository(fileUri: String): Repository? {
    val repo = gitRepositoryManger.getRepositoryForFile(virtualFileManager.findFileByUrl(fileUri)) ?: return null
    return Repository(
      root = repo.root.url,
      remotes = repo.remotes.mapNotNull { remote ->
        remote.firstUrl?.let {
          Repository.Remote(
            name = remote.name,
            url = it,
          )
        }
      }
    )
  }

  override fun diff(rootUri: String, cached: Boolean): List<String>? {
    val root = virtualFileManager.findFileByUrl(rootUri) ?: return null
    val handler = GitLineHandler(project, root, GitCommand.DIFF).apply {
      if (cached) {
        addParameters("--cached")
      }
    }
    val output = git.runCommand(handler).output
    // FIXME: sort the diffs
    return output.joinToString("\n").split(Regex("\\n(?=diff)"))
  }
}