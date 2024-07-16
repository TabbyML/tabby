package com.tabbyml.tabby4eclipse;

import org.eclipse.core.resources.IProject;
import org.eclipse.ui.IStartup;
import org.eclipse.lsp4e.LanguageServersRegistry;
import org.eclipse.lsp4e.LanguageServerWrapper;
import org.eclipse.lsp4e.LanguageServersRegistry.LanguageServerDefinition;

public class Startup implements IStartup {
	public static final String LANGUAGE_SERVER_ID = "com.tabbyml.tabby4eclipse.languageServer";

	private Logger logger = new Logger("Startup");
	
	@Override
	public void earlyStartup() {
		try {
			LanguageServerDefinition def = LanguageServersRegistry.getInstance().getDefinition(LANGUAGE_SERVER_ID);
			LanguageServerWrapper serverWrapper = new LanguageServerWrapper((IProject)null, def);
	        serverWrapper.start();
		} catch (Exception e) {
			logger.error("Failed to start Tabby language server.", e);
		}
	}

}
