package com.tabbyml.tabby4eclipse.chat;

import java.util.List;

import com.google.gson.annotations.SerializedName;

public class ChatMessage {
	private String message;
	private FileContext selectContext;
	private List<FileContext> relevantContext;
	private FileContext activeContext;

	public ChatMessage() {
	}

	public ChatMessage(String message) {
		this.message = message;
	}

	public String getMessage() {
		return message;
	}

	public void setMessage(String message) {
		this.message = message;
	}

	public FileContext getSelectContext() {
		return selectContext;
	}

	public void setSelectContext(FileContext selectContext) {
		this.selectContext = selectContext;
	}

	public List<FileContext> getRelevantContext() {
		return relevantContext;
	}

	public void setRelevantContext(List<FileContext> relevantContext) {
		this.relevantContext = relevantContext;
	}

	public FileContext getActiveContext() {
		return activeContext;
	}

	public void setActiveContext(FileContext activeContext) {
		this.activeContext = activeContext;
	}

	public static class FileContext {
		@SuppressWarnings("unused")
		private String kind = "file";
		private LineRange range;
		private String filepath;
		private String content;
		@SerializedName("git_url")
		private String gitUrl;

		public FileContext() {
		}

		public LineRange getRange() {
			return range;
		}

		public void setRange(LineRange range) {
			this.range = range;
		}

		public String getFilePath() {
			return filepath;
		}

		public void setFilePath(String filepath) {
			this.filepath = filepath;
		}

		public String getContent() {
			return content;
		}

		public void setContent(String content) {
			this.content = content;
		}

		public String getGitUrl() {
			return gitUrl;
		}

		public void setGitUrl(String gitUrl) {
			this.gitUrl = gitUrl;
		}

		public static class LineRange {
			private int start;
			private int end;

			public LineRange(int start, int end) {
				this.start = start;
				this.end = end;
			}

			public int getStart() {
				return start;
			}

			public int getEnd() {
				return end;
			}
		}
	}

}
