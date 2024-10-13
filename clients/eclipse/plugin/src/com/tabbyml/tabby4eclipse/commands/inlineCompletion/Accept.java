package com.tabbyml.tabby4eclipse.commands.inlineCompletion;

import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;

import com.tabbyml.tabby4eclipse.Logger;
import com.tabbyml.tabby4eclipse.inlineCompletion.InlineCompletionService;

public class Accept extends AbstractHandler {
	private Logger logger = new Logger("Commands.InlineCompletion.Accept");

	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		logger.debug("Accept the current inline completion.");
		InlineCompletionService.getInstance().accept();
		return null;
	}

	@Override
	public boolean isEnabled() {
		return InlineCompletionService.getInstance().isCompletionItemVisible();
	}

}
