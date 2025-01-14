package com.tabbyml.tabby4eclipse.chat;

public class PositionRange extends Range {
	private final Position start;
	private final Position end;

	public PositionRange(Position start, Position end) {
		this.start = start;
		this.end = end;
	}

	public Position getStart() {
		return start;
	}

	public Position getEnd() {
		return end;
	}
}
