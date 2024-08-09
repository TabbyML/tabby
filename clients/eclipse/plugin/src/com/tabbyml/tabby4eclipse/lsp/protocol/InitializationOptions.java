package com.tabbyml.tabby4eclipse.lsp.protocol;

public class InitializationOptions {
	private ClientProvidedConfig config;
	private ClientInfo clientInfo;
	private ClientCapabilities clientCapabilities;

	public InitializationOptions() {
	}

	public InitializationOptions(ClientProvidedConfig config, ClientInfo clientInfo,
			ClientCapabilities clientCapabilities) {
		this.config = config;
		this.clientInfo = clientInfo;
		this.clientCapabilities = clientCapabilities;
	}

	public ClientProvidedConfig getConfig() {
		return config;
	}

	public void setConfig(ClientProvidedConfig config) {
		this.config = config;
	}

	public ClientInfo getClientInfo() {
		return clientInfo;
	}

	public void setClientInfo(ClientInfo clientInfo) {
		this.clientInfo = clientInfo;
	}

	public ClientCapabilities getClientCapabilities() {
		return clientCapabilities;
	}

	public void setClientCapabilities(ClientCapabilities clientCapabilities) {
		this.clientCapabilities = clientCapabilities;
	}
}
