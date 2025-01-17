package com.tabbyml.tabby4eclipse.chat;

public class EditorFileContext {
	private final String kind;
	private final Filepath filepath;
	private final Range range;
	private final String content;

	public EditorFileContext(Filepath filepath, Range range, String content) {
		this.kind = "file";
		this.filepath = filepath;
		this.range = range;
		this.content = content;
	}

	public String getKind() {
		return kind;
	}

	public Filepath getFilepath() {
		return filepath;
	}

	public Range getRange() {
		return range;
	}

	public String getContent() {
		return content;
	}
}
