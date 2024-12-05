package com.tabbyml.tabby4eclipse.lsp.protocol;

import org.eclipse.lsp4j.Position;
import org.eclipse.lsp4j.Range;
import org.eclipse.lsp4j.TextDocumentIdentifier;

public class InlineCompletionParams {
	private InlineCompletionContext context;
	private TextDocumentIdentifier textDocument;
	private Position position;

	public InlineCompletionParams(InlineCompletionContext context, TextDocumentIdentifier textDocument,
			Position position) {
		this.context = context;
		this.textDocument = textDocument;
		this.position = position;
	}

	public InlineCompletionContext getContext() {
		return context;
	}

	public void setContext(InlineCompletionContext context) {
		this.context = context;
	}

	public TextDocumentIdentifier getTextDocument() {
		return textDocument;
	}

	public void setTextDocument(TextDocumentIdentifier textDocument) {
		this.textDocument = textDocument;
	}

	public Position getPosition() {
		return position;
	}

	public void setPosition(Position position) {
		this.position = position;
	}

	public static class InlineCompletionContext {
		private InlineCompletionTriggerKind triggerKind;
		private SelectedCompletionInfo selectedCompletionInfo;

		public InlineCompletionContext(InlineCompletionTriggerKind triggerKind,
				SelectedCompletionInfo selectedCompletionInfo) {
			this.triggerKind = triggerKind;
			this.selectedCompletionInfo = selectedCompletionInfo;
		}

		public InlineCompletionTriggerKind getTriggerKind() {
			return triggerKind;
		}

		public void setTriggerKind(InlineCompletionTriggerKind triggerKind) {
			this.triggerKind = triggerKind;
		}

		public SelectedCompletionInfo getSelectedCompletionInfo() {
			return selectedCompletionInfo;
		}

		public void setSelectedCompletionInfo(SelectedCompletionInfo selectedCompletionInfo) {
			this.selectedCompletionInfo = selectedCompletionInfo;
		}
	}

	public enum InlineCompletionTriggerKind {
		Invoked(0), Automatic(1);

		private final int value;

		InlineCompletionTriggerKind(int value) {
			this.value = value;
		}

		public int getValue() {
			return value;
		}
	}

	public static class SelectedCompletionInfo {
		private String text;
		private Range range;

		public SelectedCompletionInfo(String text, Range range) {
			this.text = text;
			this.range = range;
		}

		public String getText() {
			return text;
		}

		public void setText(String text) {
			this.text = text;
		}

		public Range getRange() {
			return range;
		}

		public void setRange(Range range) {
			this.range = range;
		}
	}
}
