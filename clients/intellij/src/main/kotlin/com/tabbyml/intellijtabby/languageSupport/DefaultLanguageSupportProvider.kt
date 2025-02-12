package com.tabbyml.intellijtabby.languageSupport

import com.intellij.codeInsight.TargetElementUtil
import com.intellij.openapi.application.ReadAction
import com.intellij.openapi.project.Project
import com.intellij.psi.PsiElement
import com.intellij.psi.PsiRecursiveElementWalkingVisitor
import com.intellij.util.concurrency.AppExecutorUtil
import com.tabbyml.intellijtabby.findEditor
import com.tabbyml.intellijtabby.languageSupport.LanguageSupportProvider.*
import org.eclipse.lsp4j.SemanticTokenTypes
import java.util.concurrent.Callable
import java.util.concurrent.CompletableFuture

/**
 * The default implementation of [LanguageSupportProvider].
 * This implementation relies on [TargetElementUtil] and tries to find the navigation target at each position in the
 * editor to provide semantic tokens and declarations.
 * This implementation may not work effectively for all languages.
 */
open class DefaultLanguageSupportProvider : LanguageSupportProvider {
  private val targetElementUtil = TargetElementUtil.getInstance()
  private val executor = AppExecutorUtil.getAppExecutorService()

  override fun provideSemanticTokensRange(
    project: Project,
    fileRange: FileRange
  ): CompletableFuture<List<SemanticToken>?> {
    val psiFile = fileRange.file
    val editor = project.findEditor(psiFile.virtualFile) ?: return CompletableFuture.completedFuture(null)

    val future = CompletableFuture<List<SemanticToken>?>()

    ReadAction.nonBlocking(Callable {
      val leafElements = mutableListOf<PsiElement>()
      psiFile.accept(object : PsiRecursiveElementWalkingVisitor() {
        override fun visitElement(element: PsiElement) {
          if (future.isCancelled) {
            return
          }
          if (element.children.isEmpty() &&
            element.text.matches(Regex("\\w+")) &&
            fileRange.range.contains(element.textRange) &&
            leafElements.none { it.textRange.intersects(element.textRange) }
          ) {
            leafElements.add(element)
          }
          if (element.textRange.intersects(fileRange.range)) {
            super.visitElement(element)
          }
        }
      })

      if (future.isCancelled) {
        return@Callable
      }

      val result = leafElements.mapNotNull {
        if (future.isCancelled) {
          return@mapNotNull null
        }
        val target = try {
          targetElementUtil.findTargetElement(
            editor.editor,
            TargetElementUtil.ELEMENT_NAME_ACCEPTED or TargetElementUtil.REFERENCED_ELEMENT_ACCEPTED,
            it.textRange.startOffset
          )
        } catch (e: Exception) {
          null
        }
        if (target == it || target == null || target.text == null) {
          null
        } else {
          SemanticToken(
            text = it.text,
            range = it.textRange,
            type = SemanticTokenTypes.Type, // Default to use `Type` as the token type as we don't know the actual type
          )
        }
      }

      future.complete(result)
    }).inSmartMode(project).submit(executor)

    return future
  }

  override fun provideDeclaration(project: Project, filePosition: FilePosition): CompletableFuture<List<FileRange>?> {
    val psiFile = filePosition.file
    val editor = project.findEditor(psiFile.virtualFile) ?: return CompletableFuture.completedFuture(null)

    val future = CompletableFuture<List<FileRange>?>()

    ReadAction.nonBlocking(Callable {
      val target = try {
        targetElementUtil.findTargetElement(
          editor.editor,
          TargetElementUtil.ELEMENT_NAME_ACCEPTED or TargetElementUtil.REFERENCED_ELEMENT_ACCEPTED,
          filePosition.offset
        )
      } catch (e: Exception) {
        null
      }
      val file = target?.containingFile
      val range = target?.textRange
      if (file == null || range == null) {
        future.complete(listOf())
      } else {
        future.complete(listOf(FileRange(file, range)))
      }
    }).inSmartMode(project).submit(executor)

    return future
  }
}
