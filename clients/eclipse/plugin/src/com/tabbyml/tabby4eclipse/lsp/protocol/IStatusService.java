package com.tabbyml.tabby4eclipse.lsp.protocol;

import java.util.concurrent.CompletableFuture;

import org.eclipse.lsp4j.jsonrpc.services.JsonRequest;
import org.eclipse.lsp4j.jsonrpc.services.JsonSegment;

@JsonSegment("tabby")
public interface IStatusService {
	@JsonRequest("status")
	CompletableFuture<StatusInfo> getStatus(StatusRequestParams params);

	@JsonRequest("status/showHelpMessage")
	CompletableFuture<Boolean> showHelpMessage(Object params);

	@JsonRequest("status/ignoredIssues/edit")
	CompletableFuture<Boolean> editIngoredIssues(StatusIgnoredIssuesEditParams params);
}
