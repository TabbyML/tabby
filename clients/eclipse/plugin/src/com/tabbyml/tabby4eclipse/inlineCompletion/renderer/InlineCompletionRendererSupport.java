package com.tabbyml.tabby4eclipse.inlineCompletion.renderer;

import org.eclipse.jface.text.ITextViewer;

import com.tabbyml.tabby4eclipse.inlineCompletion.InlineCompletionItem;

public class InlineCompletionRendererSupport {
	private IInlineCompletionItemRenderer renderer = InlineCompletionItemRenderer.getInstance();
	private ITextViewer currentTextViewer = null;
	private InlineCompletionItem currentCompletionItem = null;
	private String currentViewId = null;
	private Long currentDisplayAt = null;

	public void show(ITextViewer viewer, InlineCompletionItem item) {
		if (currentTextViewer != null) {
			renderer.updateInlineCompletionItem(currentTextViewer, null);
		}
		currentTextViewer = viewer;
		currentCompletionItem = item;
		renderer.updateInlineCompletionItem(viewer, item);

		currentDisplayAt = System.currentTimeMillis();

		String completionId = "no-cmpl-id";
		if (item.getEventId() != null && item.getEventId().getCompletionId() != null) {
			completionId = item.getEventId().getCompletionId().replace("cmpl-", "");
		}
		currentViewId = String.format("view-%s-at-%d", completionId, currentDisplayAt);
	}

	public void hide() {
		if (currentTextViewer != null) {
			renderer.updateInlineCompletionItem(currentTextViewer, null);
			currentTextViewer = null;
		}
		currentCompletionItem = null;
		currentViewId = null;
		currentDisplayAt = null;
	}

	public InlineCompletionItem getCurrentCompletionItem() {
		return currentCompletionItem;
	}

	public ITextViewer getCurrentTextViewer() {
		return currentTextViewer;
	}

	public String getCurrentViewId() {
		return currentViewId;
	}

	public Long getCurrentDisplayedTime() {
		if (currentDisplayAt != null) {
			return System.currentTimeMillis() - currentDisplayAt;
		}
		return null;
	}
}
