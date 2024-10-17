package com.tabbyml.tabby4eclipse.commands.inlineCompletion;

import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;

import com.tabbyml.tabby4eclipse.Logger;
import com.tabbyml.tabby4eclipse.inlineCompletion.InlineCompletionService;

public class Previous extends AbstractHandler {
	private Logger logger = new Logger("Commands.InlineCompletion.Next");

	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		logger.debug("Cycle inline completion choices: Previous.");
		InlineCompletionService.getInstance().previous();
		return null;
	}

	@Override
	public boolean isEnabled() {
		return InlineCompletionService.getInstance().isCompletionItemVisible();
	}

}
