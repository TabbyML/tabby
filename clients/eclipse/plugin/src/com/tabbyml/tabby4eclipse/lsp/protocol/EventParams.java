package com.tabbyml.tabby4eclipse.lsp.protocol;

public class EventParams {
	private String type;
	private String selectKind;
	private CompletionEventId eventId;
	private String viewId;
	private Long elapsed;

	public EventParams() {
	}

	public String getType() {
		return type;
	}

	public void setType(String type) {
		this.type = type;
	}

	public String getSelectKind() {
		return selectKind;
	}

	public void setSelectKind(String selectKind) {
		this.selectKind = selectKind;
	}

	public CompletionEventId getCompletionEventId() {
		return eventId;
	}

	public void setCompletionEventId(CompletionEventId completionEventId) {
		this.eventId = completionEventId;
	}

	public String getViewId() {
		return viewId;
	}

	public void setViewId(String viewId) {
		this.viewId = viewId;
	}

	public Long getElapsed() {
		return elapsed;
	}

	public void setElapsed(Long elapsed) {
		this.elapsed = elapsed;
	}

	public static class Type {
		public static final String VIEW = "view";
		public static final String SELECT = "select";
		public static final String DISMISS = "dismiss";
	}

	public static class SelectKind {
		public static final String LINE = "line";
	}
}
