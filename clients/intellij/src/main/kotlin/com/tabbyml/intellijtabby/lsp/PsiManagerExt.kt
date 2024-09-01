package com.tabbyml.intellijtabby.lsp

import com.intellij.openapi.application.ReadAction
import com.intellij.openapi.vfs.VirtualFile
import com.intellij.psi.PsiFile
import com.intellij.psi.PsiManager

fun PsiManager.findFileWithReadLock(virtualFile: VirtualFile): PsiFile? {
  return ReadAction.compute<PsiFile?, Throwable> { findFile(virtualFile) }
}