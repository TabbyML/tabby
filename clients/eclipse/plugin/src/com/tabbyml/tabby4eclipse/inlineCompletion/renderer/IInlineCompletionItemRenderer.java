package com.tabbyml.tabby4eclipse.inlineCompletion.renderer;

import org.eclipse.jface.text.ITextViewer;

import com.tabbyml.tabby4eclipse.inlineCompletion.InlineCompletionItem;

public interface IInlineCompletionItemRenderer {
	/**
	 * Update the inline completion item in the viewer at the current caret
	 * position.
	 * 
	 * @param textViewer The viewer to show the inline completion item in.
	 * @param item       The inline completion item to show.
	 */
	public abstract void updateInlineCompletionItem(ITextViewer textViewer, InlineCompletionItem item);
}
