package com.tabbyml.tabby4eclipse.chat;

public class FilepathInWorkspace extends Filepath {
	private final String filepath;
	private final String baseDir;

	public FilepathInWorkspace(String filepath, String baseDir) {
		super(Kind.WORKSPACE);
		this.filepath = filepath;
		this.baseDir = baseDir;
	}

	public String getFilepath() {
		return filepath;
	}

	public String getBaseDir() {
		return baseDir;
	}
}
