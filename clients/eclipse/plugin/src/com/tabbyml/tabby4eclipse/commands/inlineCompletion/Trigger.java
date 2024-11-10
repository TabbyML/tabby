package com.tabbyml.tabby4eclipse.commands.inlineCompletion;

import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;

import com.tabbyml.tabby4eclipse.Logger;
import com.tabbyml.tabby4eclipse.editor.EditorUtils;
import com.tabbyml.tabby4eclipse.inlineCompletion.InlineCompletionService;

public class Trigger extends AbstractHandler {
	private Logger logger = new Logger("Commands.InlineCompletion.Trigger");

	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		logger.debug("Trigger inline completion manually.");
		InlineCompletionService.getInstance().trigger(true);
		return null;
	}

	@Override
	public boolean isEnabled() {
		return EditorUtils.getActiveTextEditor().isEditable();
	}

}
