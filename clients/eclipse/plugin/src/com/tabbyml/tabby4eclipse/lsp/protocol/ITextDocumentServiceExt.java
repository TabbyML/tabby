package com.tabbyml.tabby4eclipse.lsp.protocol;

import java.util.concurrent.CompletableFuture;

import org.eclipse.lsp4j.jsonrpc.services.JsonRequest;
import org.eclipse.lsp4j.jsonrpc.services.JsonSegment;

@JsonSegment("textDocument")
public interface ITextDocumentServiceExt {
	@JsonRequest
	CompletableFuture<InlineCompletionList> inlineCompletion(InlineCompletionParams params);
}
