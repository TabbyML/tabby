package com.tabbyml.tabby4eclipse.lsp.protocol;

public class ClientInfo {
	private String name;
	private String version;
	private TabbyPluginInfo tabbyPlugin;

	public ClientInfo() {
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public String getVersion() {
		return version;
	}

	public void setVersion(String version) {
		this.version = version;
	}

	public TabbyPluginInfo getTabbyPlugin() {
		return tabbyPlugin;
	}

	public void setTabbyPlugin(TabbyPluginInfo tabbyPlugin) {
		this.tabbyPlugin = tabbyPlugin;
	}

	public static class TabbyPluginInfo {
		private String name;
		private String version;

		public String getName() {
			return name;
		}

		public void setName(String name) {
			this.name = name;
		}

		public String getVersion() {
			return version;
		}

		public void setVersion(String version) {
			this.version = version;
		}
	}
}