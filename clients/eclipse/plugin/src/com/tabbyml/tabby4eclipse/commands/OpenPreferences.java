package com.tabbyml.tabby4eclipse.commands;

import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;

import com.tabbyml.tabby4eclipse.preferences.MainPreferencesPage;

public class OpenPreferences extends AbstractHandler {

	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		MainPreferencesPage.openPreferences();
		return null;
	}
}
