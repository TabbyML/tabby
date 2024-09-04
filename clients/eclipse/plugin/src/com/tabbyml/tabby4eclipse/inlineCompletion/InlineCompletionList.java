package com.tabbyml.tabby4eclipse.inlineCompletion;

import java.util.List;

public class InlineCompletionList {
	private boolean isIncomplete;
	private List<InlineCompletionItem> items;

	public InlineCompletionList(boolean isIncomplete, List<InlineCompletionItem> items) {
		this.isIncomplete = isIncomplete;
		this.items = items;
	}

	public boolean isIncomplete() {
		return isIncomplete;
	}

	public List<InlineCompletionItem> getItems() {
		return items;
	}
}
