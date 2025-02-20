package com.tabbyml.tabby4eclipse.chat;

public abstract class Filepath {
	private final String kind;

	protected Filepath(String kind) {
		this.kind = kind;
	}

	public String getKind() {
		return kind;
	}

	public static class Kind {
		public static final String GIT = "git";
		public static final String WORKSPACE = "workspace";
		public static final String URI = "uri";
	}
}
