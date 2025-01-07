package com.tabbyml.tabby4eclipse.chat;

public class Position {
	private final int line;
	private final int character;

	public Position(int line, int character) {
		this.line = line;
		this.character = character;
	}

	public int getLine() {
		return line;
	}

	public int getCharacter() {
		return character;
	}
}
