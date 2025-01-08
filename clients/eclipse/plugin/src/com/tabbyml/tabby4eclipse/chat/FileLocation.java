package com.tabbyml.tabby4eclipse.chat;

public class FileLocation {
	private final Filepath filepath;
	private final Object location;

	public FileLocation(Filepath filepath, Object location) {
		this.filepath = filepath;
		this.location = location;
	}

	public Filepath getFilepath() {
		return filepath;
	}

	public Object getLocation() {
		return location;
	}
}
