package com.tabbyml.tabby4eclipse.lsp;

import java.util.ArrayList;
import java.util.List;

import com.tabbyml.tabby4eclipse.lsp.protocol.Config;

public class ServerConfigHolder {
	public static ServerConfigHolder getInstance() {
		return LazyHolder.INSTANCE;
	}

	private static class LazyHolder {
		private static final ServerConfigHolder INSTANCE = new ServerConfigHolder();
	}

	private Config config = new Config();
	private List<ConfigDidChangeListener> listeners = new ArrayList<>();

	public Config getConfig() {
		return config;
	}

	public void setConfig(Config config) {
		this.config = config;
		for (ConfigDidChangeListener listener : listeners) {
			listener.configDidChange();
		}
	}

	public void addConfigDidChangeListener(ConfigDidChangeListener listener) {
		listeners.add(listener);
	}

	public interface ConfigDidChangeListener {
		void configDidChange();
	}
}
