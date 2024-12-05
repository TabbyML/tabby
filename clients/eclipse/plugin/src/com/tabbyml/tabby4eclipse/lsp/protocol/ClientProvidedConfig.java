package com.tabbyml.tabby4eclipse.lsp.protocol;

public class ClientProvidedConfig {
	private ServerConfig server;
	private InlineCompletionConfig inlineCompletion;
	private String keybindings;
	private AnonymousUsageTrackingConfig anonymousUsageTracking;

	public ClientProvidedConfig() {
	}

	public ServerConfig getServer() {
		return server;
	}

	public void setServer(ServerConfig server) {
		this.server = server;
	}

	public InlineCompletionConfig getInlineCompletion() {
		return inlineCompletion;
	}

	public void setInlineCompletion(InlineCompletionConfig inlineCompletion) {
		this.inlineCompletion = inlineCompletion;
	}

	public String getKeybindings() {
		return keybindings;
	}

	public void setKeybindings(String keybindings) {
		this.keybindings = keybindings;
	}

	public AnonymousUsageTrackingConfig getAnonymousUsageTracking() {
		return anonymousUsageTracking;
	}

	public void setAnonymousUsageTracking(AnonymousUsageTrackingConfig anonymousUsageTracking) {
		this.anonymousUsageTracking = anonymousUsageTracking;
	}

	public static class ServerConfig {
		private String endpoint;
		private String token;

		public ServerConfig(String endpoint, String token) {
			this.endpoint = endpoint;
			this.token = token;
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
	}

	public static class InlineCompletionConfig {
		private String triggerMode;

		public InlineCompletionConfig(String triggerMode) {
			this.triggerMode = triggerMode;
		}

		public String getTriggerMode() {
			return triggerMode;
		}

		public void setTriggerMode(String triggerMode) {
			this.triggerMode = triggerMode;
		}

		public static class TriggerMode {
			public static final String AUTO = "auto";
			public static final String MANUAL = "manual";
		}
	}

	public static class Keybindings {
		public static final String DEFAULT = "default";
		public static final String TABBY_STYLE = "tabby-style";
		public static final String CUSTOMIZE = "customize";
	}

	public static class AnonymousUsageTrackingConfig {
		private boolean disable;

		public AnonymousUsageTrackingConfig(boolean disable) {
			this.disable = disable;
		}

		public boolean getDisable() {
			return disable;
		}

		public void setDisable(boolean disable) {
			this.disable = disable;
		}
	}
}
