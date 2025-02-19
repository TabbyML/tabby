package com.tabbyml.tabby4eclipse.lsp;

import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

import org.eclipse.jface.text.IDocument;
import org.eclipse.lsp4e.LSPEclipseUtils;
import org.eclipse.lsp4j.ConfigurationParams;
import org.eclipse.lsp4j.DeclarationParams;
import org.eclipse.lsp4j.Location;
import org.eclipse.lsp4j.LocationLink;
import org.eclipse.lsp4j.Range;
import org.eclipse.lsp4j.SemanticTokensRangeParams;
import org.eclipse.lsp4j.jsonrpc.messages.Either;
import org.eclipse.lsp4j.jsonrpc.services.JsonNotification;
import org.eclipse.lsp4j.jsonrpc.services.JsonRequest;

import com.tabbyml.tabby4eclipse.git.GitProvider;
import com.tabbyml.tabby4eclipse.lsp.protocol.Config;
import com.tabbyml.tabby4eclipse.lsp.protocol.GitDiffParams;
import com.tabbyml.tabby4eclipse.lsp.protocol.GitDiffResult;
import com.tabbyml.tabby4eclipse.lsp.protocol.GitRepository;
import com.tabbyml.tabby4eclipse.lsp.protocol.GitRepositoryParams;
import com.tabbyml.tabby4eclipse.lsp.protocol.ReadFileParams;
import com.tabbyml.tabby4eclipse.lsp.protocol.ReadFileResult;
import com.tabbyml.tabby4eclipse.lsp.protocol.SemanticTokensRangeResult;
import com.tabbyml.tabby4eclipse.lsp.protocol.StatusInfo;
import com.tabbyml.tabby4eclipse.preferences.PreferencesService;

public class LanguageClientImpl extends org.eclipse.lsp4e.LanguageClientImpl {
	@Override
	public CompletableFuture<List<Object>> configuration(ConfigurationParams configurationParams) {

		return CompletableFuture.completedFuture(new ArrayList<>() {
			{
				add(PreferencesService.getInstance().buildClientProvidedConfig());
			}
		});
	}

	@JsonNotification("tabby/status/didChange")
	void statusDidChange(StatusInfo params) {
		StatusInfoHolder.getInstance().setStatusInfo(params);
	}

	@JsonNotification("tabby/config/didChange")
	void statusDidChange(Config params) {
		ServerConfigHolder.getInstance().setConfig(params);
	}

	@JsonRequest("tabby/workspaceFileSystem/readFile")
	CompletableFuture<ReadFileResult> workspaceReadFile(ReadFileParams params) {
		if (params.getFormat().equals("text")) {
			try {
				URI uri = new URI(params.getUri());
				IDocument document = LSPEclipseUtils.getDocument(uri);
				Range range = params.getRange();

				ReadFileResult result = new ReadFileResult();
				if (range != null) {
					int start = LSPEclipseUtils.toOffset(range.getStart(), document);
					int end = LSPEclipseUtils.toOffset(range.getEnd(), document);
					result.setText(document.get(start, end - start));
				} else {
					result.setText(document.get());
				}
				return CompletableFuture.completedFuture(result);
			} catch (Exception e) {
				return CompletableFuture.completedFuture(null);
			}
		} else {
			return CompletableFuture.completedFuture(null);
		}
	}

	@JsonRequest("tabby/git/repository")
	CompletableFuture<GitRepository> gitRepository(GitRepositoryParams params) {
		CompletableFuture<GitRepository> future = new CompletableFuture<>();
		GitRepository result = GitProvider.getInstance().getRepository(params);
		future.complete(result);
		return future;
	}

	@JsonRequest("tabby/git/diff")
	CompletableFuture<GitDiffResult> gitDiff(GitDiffParams params) {
		CompletableFuture<GitDiffResult> future = new CompletableFuture<>();
		GitDiffResult result = GitProvider.getInstance().getDiff(params);
		future.complete(result);
		return future;
	}

	@JsonRequest("tabby/languageSupport/textDocument/declaration")
	CompletableFuture<Either<List<? extends Location>, List<? extends LocationLink>>> languageSupportDeclaration(
			DeclarationParams params) {
		return LanguageSupportProvider.getInstance().languageSupportDeclaration(params);
	}

	@JsonRequest("tabby/languageSupport/textDocument/semanticTokens/range")
	CompletableFuture<SemanticTokensRangeResult> languageSupportSemanticTokensRange(SemanticTokensRangeParams params) {
		return LanguageSupportProvider.getInstance().languageSupportSemanticTokensRange(params);
	}
}
