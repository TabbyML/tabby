package com.tabbyml.tabby4eclipse.commands.inlineCompletion;

import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;

import com.tabbyml.tabby4eclipse.Logger;
import com.tabbyml.tabby4eclipse.inlineCompletion.InlineCompletionService;

public class Dismiss extends AbstractHandler {
	private Logger logger = new Logger("Commands.InlineCompletion.Dismiss");
	
	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		logger.debug("Dismiss the current inline completion.");
		InlineCompletionService.getInstance().dismiss();
		return null;
	}

	@Override
	public boolean isEnabled() {
		return InlineCompletionService.getInstance().isCompletionItemVisible();
	}

}
