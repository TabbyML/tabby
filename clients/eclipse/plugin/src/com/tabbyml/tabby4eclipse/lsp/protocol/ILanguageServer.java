package com.tabbyml.tabby4eclipse.lsp.protocol;

import org.eclipse.lsp4j.jsonrpc.services.JsonDelegate;
import org.eclipse.lsp4j.services.LanguageServer;

public interface ILanguageServer extends LanguageServer {
	@JsonDelegate
	ITextDocumentServiceExt getTextDocumentServiceExt();

	@JsonDelegate
	IStatusService getStatusService();
	
	@JsonDelegate
	ITelemetryService getTelemetryService();
}
