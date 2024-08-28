package com.tabbyml.tabby4eclipse.lsp.protocol;

import org.eclipse.lsp4j.Range;

public class ReadFileParams {
	private String uri;
	private String format;
	private Range range;

	public ReadFileParams(String uri) {
		this.uri = uri;
	}

	public String getUri() {
		return uri;
	}

	public void setUri(String uri) {
		this.uri = uri;
	}

	public String getFormat() {
		return format;
	}

	public void setFormat(String format) {
		this.format = format;
	}

	public Range getRange() {
		return range;
	}

	public void setRange(Range range) {
		this.range = range;
	}
}
