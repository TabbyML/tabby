package com.tabbyml.tabby4eclipse.lsp.protocol;

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

	public void setIncomplete(boolean isIncomplete) {
		this.isIncomplete = isIncomplete;
	}

	public List<InlineCompletionItem> getItems() {
		return items;
	}

	public void setItems(List<InlineCompletionItem> items) {
		this.items = items;
	}
}
