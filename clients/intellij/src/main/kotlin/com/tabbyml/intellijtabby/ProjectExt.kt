package com.tabbyml.intellijtabby

import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.application.runReadAction
import com.intellij.openapi.editor.Document
import com.intellij.openapi.fileEditor.FileDocumentManager
import com.intellij.openapi.fileEditor.TextEditor
import com.intellij.openapi.fileEditor.ex.FileEditorManagerEx
import com.intellij.openapi.project.Project
import com.intellij.openapi.vfs.VirtualFile
import com.intellij.openapi.vfs.VirtualFileManager
import com.intellij.psi.PsiFile
import com.intellij.psi.PsiManager
import com.intellij.util.messages.Topic

fun <L : Any> Project.safeSyncPublisher(topic: Topic<L>): L? {
  return if (isDisposed) {
    null
  } else {
    messageBus.let {
      if (it.isDisposed) {
        null
      } else {
        it.syncPublisher(topic)
      }
    }
  }
}


fun Project.findVirtualFile(fileUri: String): VirtualFile? {
  val virtualFileManager = VirtualFileManager.getInstance()
  return virtualFileManager.findFileByUrl(fileUri)
}

fun Project.findDocument(fileUri: String): Document? {
  return findVirtualFile(fileUri)?.let { findDocument(it) }
}

fun Project.findDocument(virtualFile: VirtualFile): Document? {
  val fileDocumentManager = FileDocumentManager.getInstance()
  return runReadAction { fileDocumentManager.getDocument(virtualFile) }
}

fun Project.findPsiFile(fileUri: String): PsiFile? {
  return findVirtualFile(fileUri)?.let { findPsiFile(it) }
}

fun Project.findPsiFile(virtualFile: VirtualFile): PsiFile? {
  val psiManager = PsiManager.getInstance(this)
  return runReadAction { psiManager.findFile(virtualFile) }
}

fun Project.findEditor(fileUri: String): TextEditor? {
  return findVirtualFile(fileUri)?.let { findEditor(it) }
}

fun Project.findEditor(virtualFile: VirtualFile): TextEditor? {
  val fileEditorManager = FileEditorManagerEx.getInstanceEx(this)

  return runInEdtAndWait {
    fileEditorManager.getEditors(virtualFile)
  }.firstOrNull { editor -> editor is TextEditor } as? TextEditor?
}

private fun <T> runInEdtAndWait(runnable: () -> T): T {
  val app = ApplicationManager.getApplication()
  if (app.isDispatchThread) {
    return runnable()
  } else {
    var resultRef: T? = null
    app.invokeAndWait { resultRef = runnable() }
    @Suppress("UNCHECKED_CAST") return resultRef as T
  }
}
