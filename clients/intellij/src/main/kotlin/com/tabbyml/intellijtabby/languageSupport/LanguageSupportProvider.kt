package com.tabbyml.intellijtabby.languageSupport

import com.intellij.openapi.project.Project
import com.intellij.openapi.util.TextRange
import com.intellij.psi.PsiFile
import java.util.concurrent.CompletableFuture

interface LanguageSupportProvider {
  data class FilePosition(
    val file: PsiFile,
    val offset: Int,
  )

  data class FileRange(
    val file: PsiFile,
    val range: TextRange,
  )

  data class SemanticToken(
    val text: String,
    val range: TextRange,
    /**
     * See [org.eclipse.lsp4j.SemanticTokenTypes]
     */
    val type: String,
    /**
     * See [org.eclipse.lsp4j.SemanticTokenModifiers]
     */
    val modifiers: List<String> = emptyList(),
  )

  /**
   * Find all semantic tokens in the given [fileRange].
   * For now, this function is only used to find tokens that reference a declaration, which will be used to invoke [provideDeclaration] later.
   * So it is safe to only contain these tokens in the result, like class names, function names, etc.
   *
   * If no tokens are found, return an empty list.
   * If the provider does not support the given document, return null.
   */
  fun provideSemanticTokensRange(project: Project, fileRange: FileRange): CompletableFuture<List<SemanticToken>?> {
    return CompletableFuture.completedFuture(null)
  }

  /**
   * Get the declaration location for the token at the given [filePosition].
   * If no declaration is found, return an empty list.
   * If the provider does not support the given document, return null.
   */
  fun provideDeclaration(project: Project, filePosition: FilePosition): CompletableFuture<List<FileRange>?> {
    return CompletableFuture.completedFuture(null)
  }
}
