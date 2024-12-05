package com.tabbyml.intellijtabby.languageSupport

import com.intellij.openapi.components.Service
import com.intellij.openapi.diagnostic.logger
import com.intellij.openapi.extensions.ExtensionPointName
import com.intellij.openapi.project.Project
import com.tabbyml.intellijtabby.languageSupport.LanguageSupportProvider.*


@Service(Service.Level.PROJECT)
class LanguageSupportService(private val project: Project) {
  private val logger = logger<LanguageSupportService>()
  private val languageSupportProviderExtensionPoint: ExtensionPointName<LanguageSupportProvider> =
    ExtensionPointName.create("com.tabbyml.intellij-tabby.languageSupportProvider")
  private val defaultLanguageSupportProvider = DefaultLanguageSupportProvider()

  fun provideSemanticTokensRange(fileRange: FileRange): List<SemanticToken>? {
    var semanticTokens: List<SemanticToken>? = null
    for (provider in languageSupportProviderExtensionPoint.extensionList) {
      semanticTokens = provider.provideSemanticTokensRange(project, fileRange)
      if (semanticTokens != null) {
        logger.trace("Semantic tokens provided by ${provider.javaClass.name}: $semanticTokens")
        break
      }
    }
    if (semanticTokens == null) {
      semanticTokens = defaultLanguageSupportProvider.provideSemanticTokensRange(project, fileRange)
      logger.trace("Semantic tokens provided by default provider: $semanticTokens")
    }
    return semanticTokens
  }

  fun provideDeclaration(position: FilePosition): List<FileRange>? {
    var declaration: List<FileRange>? = null
    for (provider in languageSupportProviderExtensionPoint.extensionList) {
      declaration = provider.provideDeclaration(project, position)
      if (declaration != null) {
        logger.trace("Declaration provided by ${provider.javaClass.name}: $declaration")
        break
      }
    }
    if (declaration == null) {
      declaration = defaultLanguageSupportProvider.provideDeclaration(project, position)
      logger.trace("Declaration provided by default provider: $declaration")
    }
    return declaration
  }
}
