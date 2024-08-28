package com.tabbyml.tabby4eclipse.lsp.protocol;

public class ReadFileResult {
	private String text;

	public ReadFileResult() {
	}

	public ReadFileResult(String text) {
		this.text = text;
	}

	public String getText() {
		return text;
	}

	public void setText(String text) {
		this.text = text;
	}
}
