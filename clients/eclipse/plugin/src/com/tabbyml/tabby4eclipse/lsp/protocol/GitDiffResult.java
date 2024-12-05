package com.tabbyml.tabby4eclipse.lsp.protocol;

public class GitDiffResult {
	private String diff;

	public GitDiffResult() {
	}

	public GitDiffResult(String diff) {
		this.diff = diff;
	}

	public String getDiff() {
		return diff;
	}

	public void setDiff(String diff) {
		this.diff = diff;
	}
}
