package com.tabbyml.tabby4eclipse.lsp.protocol;

public class GitRepository {
	private String root;
	private String remoteUrl;

	public GitRepository() {
	}

	public String getRoot() {
		return root;
	}

	public void setRoot(String root) {
		this.root = root;
	}

	public String getRemoteUrl() {
		return remoteUrl;
	}

	public void setRemoteUrl(String remoteUrl) {
		this.remoteUrl = remoteUrl;
	}
}
