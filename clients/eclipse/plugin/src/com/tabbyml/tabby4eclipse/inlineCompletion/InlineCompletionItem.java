package com.tabbyml.tabby4eclipse.inlineCompletion;

public class InlineCompletionItem {

	public static class ReplaceRange {
		private int prefixLength;
		private int suffixLength;

		public ReplaceRange(int prefix, int suffix) {
			this.prefixLength = prefix;
			this.suffixLength = suffix;
		}

		public int getPrefixLength() {
			return prefixLength;
		}

		public int getSuffixLength() {
			return suffixLength;
		}
	}

	private String insertText;
	private ReplaceRange replaceRange;

	public InlineCompletionItem(String insertText, ReplaceRange replaceRange) {
		this.insertText = insertText;
		this.replaceRange = replaceRange;
	}

	public String getInsertText() {
		return insertText;
	}

	public ReplaceRange getReplaceRange() {
		return replaceRange;
	}
}
