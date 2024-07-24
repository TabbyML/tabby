package com.tabbyml.tabby4eclipse.editor;

public class InlineCompletionItem {
	
	public static class Range {
		private int start;
		private int end;

		public Range(int start, int end) {
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
	
	private String insertText;
	private Range replaceRange;

	public InlineCompletionItem(String insertText, Range replaceRange) {
		this.insertText = insertText;
		this.replaceRange = replaceRange;
	}
	
	public String getInsertText() {
		return insertText;
	}
	
	public Range getReplaceRange() {
		return replaceRange;
	}
}
