package com.tabbyml.tabby4eclipse.lsp.protocol;

public class CompletionEventId {
	private String completionId;
	private int choiceIndex;

	public CompletionEventId(String completionId, int choiceIndex) {
		this.completionId = completionId;
		this.choiceIndex = choiceIndex;
	}

	public String getCompletionId() {
		return completionId;
	}

	public void setCompletionId(String completionId) {
		this.completionId = completionId;
	}

	public int getChoiceIndex() {
		return choiceIndex;
	}

	public void setChoiceIndex(int choiceIndex) {
		this.choiceIndex = choiceIndex;
	}
}
