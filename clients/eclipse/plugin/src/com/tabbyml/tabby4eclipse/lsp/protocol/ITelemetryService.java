package com.tabbyml.tabby4eclipse.lsp.protocol;

import org.eclipse.lsp4j.jsonrpc.services.JsonNotification;
import org.eclipse.lsp4j.jsonrpc.services.JsonSegment;

@JsonSegment("tabby/telemetry")
public interface ITelemetryService {
	@JsonNotification
	void event(EventParams params);
}
