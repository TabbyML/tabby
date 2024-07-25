package com.tabbyml.tabby4eclipse.commands.inlineCompletion;

import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;

import com.tabbyml.tabby4eclipse.editor.InlineCompletionService;

public class Accept extends AbstractHandler {

	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		InlineCompletionService.getInstance().accept();
		return null;
	}

	@Override
	public boolean isEnabled() {
		return InlineCompletionService.getInstance().isInlineCompletionVisible();
	}

}
