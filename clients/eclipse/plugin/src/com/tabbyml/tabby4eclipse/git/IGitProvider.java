package com.tabbyml.tabby4eclipse.git;

import com.tabbyml.tabby4eclipse.lsp.protocol.GitDiffParams;
import com.tabbyml.tabby4eclipse.lsp.protocol.GitDiffResult;
import com.tabbyml.tabby4eclipse.lsp.protocol.GitRepository;
import com.tabbyml.tabby4eclipse.lsp.protocol.GitRepositoryParams;

public interface IGitProvider {
	/**
	 * Return false or throw NoClassDefFoundError if this provider is not available
	 */
	default public boolean isAvailable() throws NoClassDefFoundError {
		return false;
	}
	
	default public GitRepository getRepository(GitRepositoryParams params) {
		return null;
	}
	default public GitDiffResult getDiff(GitDiffParams params) {
		return null;
	}
}
