package com.tabbyml.intellijtabby.languageSupport

import com.intellij.openapi.components.Service
import com.intellij.openapi.diagnostic.logger
import com.intellij.openapi.extensions.ExtensionPointName
import com.intellij.openapi.project.Project
import com.tabbyml.intellijtabby.languageSupport.LanguageSupportProvider.*
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit


@Service(Service.Level.PROJECT)
class LanguageSupportService(private val project: Project) {
  private val logger = logger<LanguageSupportService>()
  private val languageSupportProviderExtensionPoint: ExtensionPointName<LanguageSupportProvider> =
    ExtensionPointName.create("com.tabbyml.intellij-tabby.languageSupportProvider")
  private val defaultLanguageSupportProvider = DefaultLanguageSupportProvider()

  fun provideSemanticTokensRange(fileRange: FileRange): CompletableFuture<List<SemanticToken>?> {
    val providers = languageSupportProviderExtensionPoint.extensionList.iterator()
    val future = CompletableFuture<List<SemanticToken>?>()
    future.completeOnTimeout(null, TIMEOUT_SEMANTIC_TOKENS_RANGE_PROVIDER, TimeUnit.MILLISECONDS)
    computeSemanticTokensRangeFromProviders(future, providers, fileRange)
    return future
  }

  private fun computeSemanticTokensRangeFromProviders(
    future: CompletableFuture<List<SemanticToken>?>,
    providers: Iterator<LanguageSupportProvider>,
    fileRange: FileRange
  ) {
    if (future.isDone) {
      return
    }
    val pair = if (providers.hasNext()) {
      val provider = providers.next()
      Pair(provider) { result: List<SemanticToken>? ->
        if (result != null) {
          logger.trace("Semantic tokens provided by ${provider.javaClass.name}: $result")
          future.complete(result)
        } else {
          // next provider
          computeSemanticTokensRangeFromProviders(future, providers, fileRange)
        }
      }
    } else {
      Pair(defaultLanguageSupportProvider) { result: List<SemanticToken>? ->
        if (result != null) {
          logger.trace("Semantic tokens provided by default provider: $result")
          future.complete(result)
        } else {
          future.complete(null)
        }
      }
    }

    val request = pair.first.provideSemanticTokensRange(project, fileRange)
    future.whenComplete { _, _ ->
      request.cancel(true)
    }
    request.thenAccept { result ->
      pair.second(result)
    }
  }

  fun provideDeclaration(position: FilePosition): CompletableFuture<List<FileRange>?> {
    val providers = languageSupportProviderExtensionPoint.extensionList.iterator()
    val future = CompletableFuture<List<FileRange>?>()
    future.completeOnTimeout(null, TIMEOUT_DECLARATION_PROVIDER, TimeUnit.MILLISECONDS)
    computeDeclarationFromProviders(future, providers, position)
    return future
  }

  private fun computeDeclarationFromProviders(
    future: CompletableFuture<List<FileRange>?>,
    providers: Iterator<LanguageSupportProvider>,
    position: FilePosition
  ) {
    if (future.isDone) {
      return
    }
    val pair = if (providers.hasNext()) {
      val provider = providers.next()
      Pair(provider) { result: List<FileRange>? ->
        if (result != null) {
          logger.trace("Declaration provided by ${provider.javaClass.name}: $result")
          future.complete(result)
        } else {
          // next provider
          computeDeclarationFromProviders(future, providers, position)
        }
      }
    } else {
      Pair(defaultLanguageSupportProvider) { result: List<FileRange>? ->
        if (result != null) {
          logger.trace("Declaration provided by default provider: $result")
          future.complete(result)
        } else {
          future.complete(null)
        }
      }
    }

    val request = pair.first.provideDeclaration(project, position)
    future.whenComplete { _, _ ->
      request.cancel(true)
    }
    request.thenAccept { result ->
      pair.second(result)
    }
  }

  companion object {
    private const val TIMEOUT_SEMANTIC_TOKENS_RANGE_PROVIDER = 100L // ms
    private const val TIMEOUT_DECLARATION_PROVIDER = 10L // ms
  }
}
