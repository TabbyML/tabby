package com.tabbyml.tabby4eclipse.commands.inlineCompletion;

import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;

import com.tabbyml.tabby4eclipse.Logger;
import com.tabbyml.tabby4eclipse.inlineCompletion.IInlineCompletionService.AcceptType;
import com.tabbyml.tabby4eclipse.inlineCompletion.InlineCompletionService;

public class AcceptNextWord extends AbstractHandler {
	private Logger logger = new Logger("Commands.InlineCompletion.AcceptNextLine");

	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		logger.debug("Accept next word of the current inline completion.");
		InlineCompletionService.getInstance().accept(AcceptType.NEXT_WORD);
		return null;
	}

	@Override
	public boolean isEnabled() {
		return InlineCompletionService.getInstance().isCompletionItemVisible();
	}

}
