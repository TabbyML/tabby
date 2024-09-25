package com.tabbyml.tabby4eclipse.inlineCompletion;

import org.eclipse.ui.texteditor.ITextEditor;

public interface IInlineCompletionTrigger {
	/**
	 * Register a text editor to be monitored for inline completion triggers.
	 */
	public void register(ITextEditor textEditor);

	/**
	 * Unregister a text editor from being monitored for inline completion triggers.
	 */
	public void unregister(ITextEditor textEditor);
}
