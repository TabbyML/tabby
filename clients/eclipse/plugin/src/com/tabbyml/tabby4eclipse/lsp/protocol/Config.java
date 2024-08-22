package com.tabbyml.tabby4eclipse.lsp.protocol;

import java.util.Map;

public class Config {
	private ServerConfig server;

	public Config() {
	}

	public ServerConfig getServer() {
		return server;
	}

	public void setServer(ServerConfig server) {
		this.server = server;
	}

	public static class ServerConfig {
		private String endpoint;
		private String token;
		private Map<String, Object> requestHeaders;

		public ServerConfig() {
		}

		public String getEndpoint() {
			return endpoint;
		}

		public void setEndpoint(String endpoint) {
			this.endpoint = endpoint;
		}

		public String getToken() {
			return token;
		}

		public void setToken(String token) {
			this.token = token;
		}

		public Map<String, Object> getRequestHeaders() {
			return requestHeaders;
		}

		public void setRequestHeaders(Map<String, Object> requestHeaders) {
			this.requestHeaders = requestHeaders;
		}
	}
}
