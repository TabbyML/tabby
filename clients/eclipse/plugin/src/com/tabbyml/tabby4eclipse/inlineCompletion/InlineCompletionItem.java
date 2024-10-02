package com.tabbyml.tabby4eclipse.inlineCompletion;

import com.tabbyml.tabby4eclipse.lsp.protocol.CompletionEventId;

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
	private CompletionEventId eventId;

	public InlineCompletionItem(String insertText, ReplaceRange replaceRange, CompletionEventId eventId) {
		this.insertText = insertText;
		this.replaceRange = replaceRange;
		this.eventId = eventId;
	}

	public String getInsertText() {
		return insertText;
	}

	public ReplaceRange getReplaceRange() {
		return replaceRange;
	}

	public CompletionEventId getEventId() {
		return eventId;
	}
}
