package com.tabbyml.tabby4eclipse.git;

import com.tabbyml.tabby4eclipse.Logger;

public class GitProvider {
	public static IGitProvider getInstance() {
		return LazyHolder.INSTANCE;
	}

	private static class LazyHolder {
		private static final IGitProvider INSTANCE = createInstance();
	}
	
	private static Logger logger = new Logger("GitProvider");

	public static IGitProvider createInstance() {
		try {
			IGitProvider eclipseJGitProvider = new EclipseJGitProvider();
			if (eclipseJGitProvider.isAvailable()) {
				return eclipseJGitProvider;
			}
		} catch (NoClassDefFoundError e) {
			logger.info("Eclipse JGitProvider is not available.");
		}
		return new NoOpGitProvider();
	}

}
