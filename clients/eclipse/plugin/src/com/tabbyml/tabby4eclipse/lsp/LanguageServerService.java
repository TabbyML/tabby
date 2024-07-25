package com.tabbyml.tabby4eclipse.lsp;

import org.eclipse.lsp4e.LanguageServerWrapper;
import org.eclipse.lsp4e.LanguageServersRegistry;
import org.eclipse.lsp4e.LanguageServersRegistry.LanguageServerDefinition;
import org.eclipse.lsp4e.LanguageServiceAccessor;

import com.tabbyml.tabby4eclipse.Logger;

public class LanguageServerService {
	public static final String LANGUAGE_SERVER_ID = "com.tabbyml.tabby4eclipse.languageServer";

	public static LanguageServerService getInstance() {
		return LazyHolder.INSTANCE;
	}

	private static class LazyHolder {
		private static final LanguageServerService INSTANCE = new LanguageServerService();
	}

	public static LanguageServerDefinition getLanguageServerDefinition() {
		LanguageServerDefinition def = LanguageServersRegistry.getInstance().getDefinition(LANGUAGE_SERVER_ID);
		return def;
	}

	private Logger logger = new Logger("LanguageServerService");
	private LanguageServerWrapper serverWrapper;

	public LanguageServerService() {
	}

	public void init() {
		try {
			serverWrapper = LanguageServiceAccessor.getLSWrapper(null, getLanguageServerDefinition());
		} catch (Exception e) {
			logger.error("Failed to start Tabby language server.", e);
		}
	}

	public LanguageServerWrapper getServer() {
		return serverWrapper;
	}
}
