package com.tabbyml.intellijtabby.languageSupport

import com.intellij.openapi.application.ReadAction.nonBlocking
import com.intellij.openapi.project.Project
import com.intellij.psi.PsiElement
import com.tabbyml.intellijtabby.languageSupport.LanguageSupportProvider.*
import io.ktor.util.*
import org.eclipse.lsp4j.SemanticTokenTypes
import org.jetbrains.kotlin.psi.*
import org.jetbrains.kotlin.psi.psiUtil.forEachDescendantOfType
import java.util.concurrent.Callable
import java.util.concurrent.CompletableFuture

class KotlinLanguageSupportProvider : LanguageSupportProvider {
  override fun provideSemanticTokensRange(
    project: Project,
    fileRange: FileRange
  ): CompletableFuture<List<SemanticToken>?> {
    val psiFile = fileRange.file
    if (psiFile.language.id.toUpperCasePreservingASCIIRules() != "KOTLIN") {
      return CompletableFuture.completedFuture(null)
    }

    val future = CompletableFuture<List<SemanticToken>?>()

    nonBlocking(Callable {
      val semanticTokens = mutableListOf<SemanticToken>()
      psiFile.forEachDescendantOfType<KtReferenceExpression> { element ->
        if (future.isCancelled) {
          return@forEachDescendantOfType
        }
        if (element.children.isEmpty() && fileRange.range.contains(element.textRange)) {
          val referenceTarget =
            element.references.map { it.resolve() }.firstNotNullOfOrNull { it } ?: return@forEachDescendantOfType
          val type = parseReferenceType(referenceTarget)
          semanticTokens.add(
            SemanticToken(
              text = element.text,
              range = element.textRange,
              type = type,
            )
          )
        }
      }
      future.complete(semanticTokens.toList())
    })

    return future
  }

  override fun provideDeclaration(project: Project, filePosition: FilePosition): CompletableFuture<List<FileRange>?> {
    val psiFile = filePosition.file
    if (psiFile.language.id.toUpperCasePreservingASCIIRules() != "KOTLIN") {
      return CompletableFuture.completedFuture(null)
    }

    val future = CompletableFuture<List<FileRange>?>()

    nonBlocking(Callable {
      val element = psiFile.findElementAt(filePosition.offset)
      val referenceExpression =
        element as? KtReferenceExpression ?: element?.parent as? KtReferenceExpression
      val referenceTarget =
        referenceExpression?.references?.map { it.resolve() }?.firstNotNullOfOrNull { it }

      val file = referenceTarget?.containingFile
      val range = referenceTarget?.textRange
      if (file == null || range == null) {
        future.complete(listOf())
      } else {
        future.complete(listOf(FileRange(file, range)))
      }
    })

    return future
  }

  private fun parseReferenceType(referenceTarget: PsiElement): String {
    return when (referenceTarget) {
      is KtClassOrObject -> SemanticTokenTypes.Class
      is KtFunction -> SemanticTokenTypes.Function
      is KtProperty -> SemanticTokenTypes.Property
      is KtParameter -> SemanticTokenTypes.Parameter
      is KtVariableDeclaration -> SemanticTokenTypes.Variable

      // 1. Fallback to `Type` for other kotlin element declarations
      // 2. The reference target may be declared in Java, fallback to `Type` for now
      else -> SemanticTokenTypes.Type
    }
  }
}