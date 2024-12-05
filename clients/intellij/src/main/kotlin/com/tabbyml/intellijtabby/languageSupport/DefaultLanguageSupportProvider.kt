package com.tabbyml.intellijtabby.languageSupport

import com.intellij.codeInsight.TargetElementUtil
import com.intellij.openapi.application.runReadAction
import com.intellij.openapi.diagnostic.logger
import com.intellij.openapi.project.Project
import com.intellij.psi.PsiElement
import com.intellij.psi.PsiRecursiveElementWalkingVisitor
import com.tabbyml.intellijtabby.findEditor
import com.tabbyml.intellijtabby.languageSupport.LanguageSupportProvider.*
import org.eclipse.lsp4j.SemanticTokenTypes

/**
 * The default implementation of [LanguageSupportProvider].
 * This implementation relies on [TargetElementUtil] and tries to find the navigation target at each position in the
 * editor to provide semantic tokens and declarations.
 * This implementation may not work effectively for all languages.
 */
open class DefaultLanguageSupportProvider : LanguageSupportProvider {
  private val logger = logger<DefaultLanguageSupportProvider>()
  private val targetElementUtil = TargetElementUtil.getInstance()

  override fun provideSemanticTokensRange(project: Project, fileRange: FileRange): List<SemanticToken>? {
    val psiFile = fileRange.file
    val editor = project.findEditor(psiFile.virtualFile) ?: return null

    return runReadAction {
      val leafElements = mutableListOf<PsiElement>()
      psiFile.accept(object : PsiRecursiveElementWalkingVisitor() {
        override fun visitElement(element: PsiElement) {
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

      leafElements.mapNotNull {
        val target = try {
          targetElementUtil.findTargetElement(
            editor.editor,
            TargetElementUtil.ELEMENT_NAME_ACCEPTED or TargetElementUtil.REFERENCED_ELEMENT_ACCEPTED,
            it.textRange.startOffset
          )
        } catch (e: Exception) {
          logger.debug("Failed to find target element when providing semantic tokens", e)
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
    }
  }

  override fun provideDeclaration(project: Project, filePosition: FilePosition): List<FileRange>? {
    val psiFile = filePosition.file
    val editor = project.findEditor(psiFile.virtualFile) ?: return null

    return runReadAction {
      val target = try {
        targetElementUtil.findTargetElement(
          editor.editor,
          TargetElementUtil.ELEMENT_NAME_ACCEPTED or TargetElementUtil.REFERENCED_ELEMENT_ACCEPTED,
          filePosition.offset
        )
      } catch (e: Exception) {
        logger.debug("Failed to find target element at ${psiFile.virtualFile.url}:${filePosition.offset}", e)
        null
      }
      val file = target?.containingFile ?: return@runReadAction listOf()
      val range = target.textRange
      listOf(FileRange(file, range))
    }
  }
}
