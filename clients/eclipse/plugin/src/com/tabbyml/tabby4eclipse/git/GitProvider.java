package com.tabbyml.tabby4eclipse.git;

import java.io.File;
import java.net.URI;
import java.util.List;

import org.eclipse.jgit.lib.Repository;
import org.eclipse.jgit.storage.file.FileRepositoryBuilder;
import org.eclipse.jgit.transport.RemoteConfig;

import com.tabbyml.tabby4eclipse.Logger;
import com.tabbyml.tabby4eclipse.lsp.protocol.GitRepository;

public class GitProvider {
	private static Logger logger = new Logger("GitProvider");

	public static GitRepository getRepository(URI uri) {
		try {
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
			logger.info("Failed to get repository for: " + uri);
			return null;
		}
	}
}
