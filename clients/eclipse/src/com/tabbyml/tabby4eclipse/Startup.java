package com.tabbyml.tabby4eclipse;

import org.eclipse.ui.IStartup;

import com.tabbyml.tabby4eclipse.editor.EditorListener;
import com.tabbyml.tabby4eclipse.lsp.LanguageServerService;

public class Startup implements IStartup {

	private Logger logger = new Logger("Startup");
	
	@Override
	public void earlyStartup() {
		logger.info("Running startup actions.");
		LanguageServerService lsManager = LanguageServerService.getInstance();
		lsManager.init();
		EditorListener editorManager = EditorListener.getInstance();
		editorManager.init();
		logger.info("Finished running startup actions.");
	}
}
