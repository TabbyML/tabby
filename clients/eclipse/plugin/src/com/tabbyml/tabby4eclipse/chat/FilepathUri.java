package com.tabbyml.tabby4eclipse.chat;

public class FilepathUri extends Filepath {
	private final String uri;

	public FilepathUri(String uri) {
		super(Kind.URI);
		this.uri = uri;
	}

	public String getUri() {
		return uri;
	}
}