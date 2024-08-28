package com.tabbyml.tabby4eclipse.lsp.protocol;

public class GitDiffParams {
	private String repository;
	private boolean cached;

	public GitDiffParams() {
	}

	public GitDiffParams(String repository) {
		this.repository = repository;
		this.cached = false;
	}

	public GitDiffParams(String repository, boolean cached) {
		this.repository = repository;
		this.cached = cached;
	}

	public String getRepository() {
		return repository;
	}

	public void setRepository(String repository) {
		this.repository = repository;
	}

	public boolean getCached() {
		return cached;
	}

	public void setCached(boolean cached) {
		this.cached = cached;
	}
}
