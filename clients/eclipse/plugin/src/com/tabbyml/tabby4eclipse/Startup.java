package com.tabbyml.tabby4eclipse;

import org.eclipse.ui.IStartup;

import com.tabbyml.tabby4eclipse.editor.WorkbenchPartListener;
import com.tabbyml.tabby4eclipse.lsp.LanguageServerService;
import com.tabbyml.tabby4eclipse.preferences.PreferencesService;

public class Startup implements IStartup {

	private Logger logger = new Logger("Startup");
	
	@Override
	public void earlyStartup() {
		logger.info("Running startup actions.");
		LanguageServerService lsService = LanguageServerService.getInstance();
		lsService.init();
		WorkbenchPartListener workbenchPartListener = WorkbenchPartListener.getInstance();
		workbenchPartListener.init();
		PreferencesService preferenceService = PreferencesService.getInstance();
		preferenceService.init();
		logger.info("Finished running startup actions.");
	}
}
