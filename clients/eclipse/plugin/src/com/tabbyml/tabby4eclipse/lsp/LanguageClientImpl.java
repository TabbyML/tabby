package com.tabbyml.tabby4eclipse.lsp;

import java.util.List;
import java.util.concurrent.CompletableFuture;

import org.eclipse.lsp4j.LocationLink;
import org.eclipse.lsp4j.SemanticTokensRangeParams;
import org.eclipse.lsp4j.DeclarationParams;
import org.eclipse.lsp4j.jsonrpc.services.JsonNotification;
import org.eclipse.lsp4j.jsonrpc.services.JsonRequest;

import com.tabbyml.tabby4eclipse.git.GitProvider;
import com.tabbyml.tabby4eclipse.lsp.protocol.Config;
import com.tabbyml.tabby4eclipse.lsp.protocol.GitDiffParams;
import com.tabbyml.tabby4eclipse.lsp.protocol.GitDiffResult;
import com.tabbyml.tabby4eclipse.lsp.protocol.GitRepository;
import com.tabbyml.tabby4eclipse.lsp.protocol.GitRepositoryParams;
import com.tabbyml.tabby4eclipse.lsp.protocol.SemanticTokensRangeResult;
import com.tabbyml.tabby4eclipse.lsp.protocol.StatusInfo;

public class LanguageClientImpl extends org.eclipse.lsp4e.LanguageClientImpl {
	@JsonNotification("tabby/status/didChange")
	void statusDidChange(StatusInfo params) {
		StatusInfoHolder.getInstance().setStatusInfo(params);
	}

	@JsonNotification("tabby/config/didChange")
	void statusDidChange(Config params) {
		ServerConfigHolder.getInstance().setConfig(params);
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
	CompletableFuture<List<LocationLink>> languageSupportDeclaration(DeclarationParams params) {
		return null;
	}

	@JsonRequest("tabby/languageSupport/textDocument/semanticTokens/range")
	CompletableFuture<SemanticTokensRangeResult> languageSupportSemanticTokensRange(SemanticTokensRangeParams params) {
		return null;
	}
}
