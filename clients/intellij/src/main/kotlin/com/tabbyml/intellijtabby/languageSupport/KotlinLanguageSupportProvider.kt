package com.tabbyml.intellijtabby.languageSupport

import com.intellij.openapi.application.runReadAction
import com.intellij.openapi.project.Project
import com.intellij.psi.PsiElement
import com.tabbyml.intellijtabby.languageSupport.LanguageSupportProvider.*
import io.ktor.util.*
import org.eclipse.lsp4j.SemanticTokenTypes
import org.jetbrains.kotlin.psi.*
import org.jetbrains.kotlin.psi.psiUtil.forEachDescendantOfType

class KotlinLanguageSupportProvider : LanguageSupportProvider {
  override fun provideSemanticTokensRange(project: Project, fileRange: FileRange): List<SemanticToken>? {
    val psiFile = fileRange.file
    if (psiFile.language.id.toUpperCasePreservingASCIIRules() != "KOTLIN") {
      return null
    }
    return runReadAction {
      val semanticTokens = mutableListOf<SemanticToken>()
      psiFile.forEachDescendantOfType<KtReferenceExpression> { element ->
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
      semanticTokens.toList()
    }
  }

  override fun provideDeclaration(project: Project, filePosition: FilePosition): List<FileRange>? {
    val psiFile = filePosition.file
    if (psiFile.language.id.toUpperCasePreservingASCIIRules() != "KOTLIN") {
      return null
    }
    return runReadAction {
      val element = psiFile.findElementAt(filePosition.offset)
      val referenceExpression =
        element as? KtReferenceExpression ?: element?.parent as? KtReferenceExpression ?: return@runReadAction listOf()
      val referenceTarget =
        referenceExpression.references.map { it.resolve() }.firstNotNullOfOrNull { it } ?: return@runReadAction listOf()
      val file = referenceTarget.containingFile ?: return@runReadAction listOf()
      val range = referenceTarget.textRange
      listOf(FileRange(file, range))
    }
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