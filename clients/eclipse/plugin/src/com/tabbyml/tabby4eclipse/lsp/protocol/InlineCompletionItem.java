package com.tabbyml.tabby4eclipse.lsp.protocol;

import org.eclipse.lsp4j.Command;
import org.eclipse.lsp4j.Range;

public class InlineCompletionItem {
	private String insertText;
	private String filterText;
	private Range range;
	private Command command;
	private Data data;

	public InlineCompletionItem(String insertText, String filterText, Range range, Command command, Data data) {
		this.insertText = insertText;
		this.filterText = filterText;
		this.range = range;
		this.command = command;
		this.data = data;
	}

	public String getInsertText() {
		return insertText;
	}

	public void setInsertText(String insertText) {
		this.insertText = insertText;
	}

	public String getFilterText() {
		return filterText;
	}

	public void setFilterText(String filterText) {
		this.filterText = filterText;
	}

	public Range getRange() {
		return range;
	}

	public void setRange(Range range) {
		this.range = range;
	}

	public Command getCommand() {
		return command;
	}

	public void setCommand(Command command) {
		this.command = command;
	}

	public Data getData() {
		return data;
	}

	public void setData(Data data) {
		this.data = data;
	}

	public static class Data {
		private CompletionEventId eventId;

		public Data(CompletionEventId eventId) {
			this.eventId = eventId;
		}

		public CompletionEventId getEventId() {
			return eventId;
		}

		public void setEventId(CompletionEventId eventId) {
			this.eventId = eventId;
		}
	}
}
