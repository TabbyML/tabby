package com.tabbyml.tabby4eclipse.lsp;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.net.URI;

import org.eclipse.jface.text.IDocument;
import org.eclipse.lsp4e.LSPEclipseUtils;
import org.eclipse.lsp4e.LanguageServers;
import org.eclipse.lsp4j.DeclarationParams;
import org.eclipse.lsp4j.Location;
import org.eclipse.lsp4j.LocationLink;
import org.eclipse.lsp4j.SemanticTokensLegend;
import org.eclipse.lsp4j.SemanticTokensRangeParams;
import org.eclipse.lsp4j.jsonrpc.messages.Either;

import com.tabbyml.tabby4eclipse.Logger;
import com.tabbyml.tabby4eclipse.lsp.protocol.SemanticTokensRangeResult;

public class LanguageSupportProvider {
	public static LanguageSupportProvider getInstance() {
		return LazyHolder.INSTANCE;
	}

	private static class LazyHolder {
		private static final LanguageSupportProvider INSTANCE = new LanguageSupportProvider();
	}

	private Logger logger = new Logger("LanguageSupportProvider");

	CompletableFuture<Either<List<? extends Location>, List<? extends LocationLink>>> languageSupportDeclaration(
			DeclarationParams params) {
		try {
			URI uri = new URI(params.getTextDocument().getUri());
			IDocument document = LSPEclipseUtils.getDocument(uri);

			CompletableFuture<Either<List<? extends Location>, List<? extends LocationLink>>> future = LanguageServers
					.forDocument(document).withFilter((serverCapabilities) -> {
						return serverCapabilities.getDeclarationProvider() != null;
					}).computeFirst((wrapper, server) -> {
						return server.getTextDocumentService().declaration(params);
					}).thenApply((result) -> {
						if (result.isPresent()) {
							return result.get();
						} else {
							return null;
						}
					});
			return future;
		} catch (Exception e) {
			logger.error("Failed to handle request languageSupport/declaration.", e);
			return CompletableFuture.completedFuture(null);
		}
	}

	CompletableFuture<SemanticTokensRangeResult> languageSupportSemanticTokensRange(SemanticTokensRangeParams params) {
		try {
			URI uri = new URI(params.getTextDocument().getUri());
			IDocument document = LSPEclipseUtils.getDocument(uri);

			CompletableFuture<SemanticTokensRangeResult> future = LanguageServers.forDocument(document)
					.withFilter((serverCapabilities) -> {
						return serverCapabilities.getSelectionRangeProvider() != null;
					}).computeFirst((wrapper, server) -> {
						return server.getTextDocumentService().semanticTokensRange(params).thenApply((tokens) -> {
							SemanticTokensRangeResult result = new SemanticTokensRangeResult();
							SemanticTokensLegend legend = wrapper.getServerCapabilities().getSemanticTokensProvider()
									.getLegend();
							result.setLegend(legend);
							result.setTokens(tokens);
							return result;
						});
					}).thenApply((result) -> {
						if (result.isPresent()) {
							return result.get();
						} else {
							return null;
						}
					});
			return future;
		} catch (Exception e) {
			logger.error("Failed to handle request languageSupport/semanticTokens/range.", e);
			return CompletableFuture.completedFuture(null);
		}
	}

}
