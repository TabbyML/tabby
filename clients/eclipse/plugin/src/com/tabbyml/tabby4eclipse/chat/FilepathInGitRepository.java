package com.tabbyml.tabby4eclipse.chat;

public class FilepathInGitRepository extends Filepath {
	private final String filepath;
	private final String gitUrl;
	private final String revision;

	public FilepathInGitRepository(String filepath, String gitUrl) {
		this(filepath, gitUrl, null);
	}
	
	public FilepathInGitRepository(String filepath, String gitUrl, String revision) {
		super(Kind.GIT);
		this.filepath = filepath;
		this.gitUrl = gitUrl;
		this.revision = revision;
	}

	public String getFilepath() {
		return filepath;
	}

	public String getGitUrl() {
		return gitUrl;
	}

	public String getRevision() {
		return revision;
	}
}
