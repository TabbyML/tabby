package com.tabbyml.tabby4eclipse.git;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.net.URI;
import java.util.List;

import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.lib.Repository;
import org.eclipse.jgit.storage.file.FileRepositoryBuilder;
import org.eclipse.jgit.transport.RemoteConfig;

import com.tabbyml.tabby4eclipse.Logger;
import com.tabbyml.tabby4eclipse.lsp.protocol.GitDiffParams;
import com.tabbyml.tabby4eclipse.lsp.protocol.GitDiffResult;
import com.tabbyml.tabby4eclipse.lsp.protocol.GitRepository;
import com.tabbyml.tabby4eclipse.lsp.protocol.GitRepositoryParams;

public class EclipseJGitProvider implements IGitProvider {
	private Logger logger = new Logger("EclipseJGitProvider");

	@Override
	public boolean isAvailable() {
		try {
			FileRepositoryBuilder builder = new FileRepositoryBuilder();
			return true;
		} catch (NoClassDefFoundError e) {
			logger.error("org.eclipse.jgit is not available.");
			return false;
		}
	}

	@Override
	public GitRepository getRepository(GitRepositoryParams params) {
		try {
			URI uri = new URI(params.getUri());
			File file = new File(uri);
			FileRepositoryBuilder builder = new FileRepositoryBuilder();
			Repository repository = builder.findGitDir(file).build();
			if (repository != null) {
				GitRepository gitRepository = new GitRepository();

				String root = repository.getDirectory().getParentFile().toURI().toString();
				gitRepository.setRoot(root);

				List<RemoteConfig> remotes = RemoteConfig.getAllRemoteConfigs(repository.getConfig());
				String firstRemoteUrl = !remotes.isEmpty() ? remotes.get(0).getURIs().get(0).toString() : null;
				String originRemoteUrl = null;
				String upstreamRemoteUrl = null;
				for (RemoteConfig remote : remotes) {
					if (remote.getName().equals("origin")) {
						originRemoteUrl = remote.getURIs().get(0).toString();
					} else if (remote.getName().equals("upstream")) {
						upstreamRemoteUrl = remote.getURIs().get(0).toString();
					}
				}
				if (originRemoteUrl != null) {
					gitRepository.setRemoteUrl(originRemoteUrl);
				} else if (upstreamRemoteUrl != null) {
					gitRepository.setRemoteUrl(upstreamRemoteUrl);
				} else {
					gitRepository.setRemoteUrl(firstRemoteUrl);
				}
				return gitRepository;
			} else {
				return null;
			}
		} catch (Exception e) {
			logger.warn("Failed to get repository for: " + params.getUri(), e);
			return null;
		}
	}

	@Override
	public GitDiffResult getDiff(GitDiffParams params) {
		try {
			URI repoPath = new URI(params.getRepository());
			File repoDir = new File(repoPath);
			FileRepositoryBuilder builder = new FileRepositoryBuilder();
			Repository repository = builder.findGitDir(repoDir).build();
			if (repository != null) {
				try (Git git = new Git(repository)) {
					ByteArrayOutputStream outStream = new ByteArrayOutputStream();
					git.diff().setCached(params.getCached()).setOutputStream(outStream).call();
					String diff = outStream.toString("UTF-8");

					GitDiffResult result = new GitDiffResult(diff);
					outStream.close();
					return result;
				}
			} else {
				return null;
			}
		} catch (Exception e) {
			logger.warn("Failed to get diff for: " + params.getRepository(), e);
			return null;
		}
	}
}
