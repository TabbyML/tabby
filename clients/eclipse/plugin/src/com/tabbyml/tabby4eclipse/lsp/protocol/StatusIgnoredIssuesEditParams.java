package com.tabbyml.tabby4eclipse.lsp.protocol;

public class StatusIgnoredIssuesEditParams {
	private String operation;
	private String[] issues;

	public StatusIgnoredIssuesEditParams() {
	}

	public String getOperation() {
		return operation;
	}

	public void setOperation(String operation) {
		this.operation = operation;
	}

	public String[] getIssues() {
		return issues;
	}

	public void setIssues(String[] issues) {
		this.issues = issues;
	}

	public static class Operation {
		public static final String ADD = "add";
		public static final String REMOVE = "remove";
		public static final String REMOVE_ALL = "removeAll";
	}

	public static class StatusIssuesName {
		public static final String COMPLETION_RESPONSE_SLOW = "completionResponseSlow";
	}
}
