package com.tabbyml.tabby4eclipse.lsp;

import java.util.concurrent.CompletableFuture;

import org.eclipse.lsp4j.jsonrpc.services.JsonNotification;

import com.tabbyml.tabby4eclipse.lsp.protocol.StatusInfo;
import com.tabbyml.tabby4eclipse.statusbar.StatusInfoHolder;

public class LanguageClientImpl extends org.eclipse.lsp4e.LanguageClientImpl {
	@JsonNotification("tabby/status/didChange")
	void statusDidChange(StatusInfo params) {
		StatusInfoHolder.getInstance().setStatusInfo(params);
	}
}
