package com.tabbyml.tabby4eclipse;

import org.eclipse.ui.IStartup;

import com.tabbyml.tabby4eclipse.editor.EditorManager;
import com.tabbyml.tabby4eclipse.lsp.LanguageServerManager;

public class Startup implements IStartup {

	private Logger logger = new Logger("Startup");
	
	@Override
	public void earlyStartup() {
		logger.info("Running startup actions.");
		LanguageServerManager lsManager = LanguageServerManager.getInstance();
		lsManager.init();
		EditorManager editorManager = EditorManager.getInstance();
		editorManager.init();
		logger.info("Finished running startup actions.");
	}
}
