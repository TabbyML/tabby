package com.tabbyml.tabby4eclipse.lsp.protocol;

public class GitRepositoryParams {
	private String uri;

	public GitRepositoryParams(String uri) {
		this.uri = uri;
	}

	public String getUri() {
		return uri;
	}

	public void setUri(String uri) {
		this.uri = uri;
	}
}
