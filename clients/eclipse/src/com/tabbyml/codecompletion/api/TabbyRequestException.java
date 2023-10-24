package com.tabbyml.codecompletion.api;

public class TabbyRequestException extends RuntimeException{

	private static final long serialVersionUID = 1L;

	public TabbyRequestException() {
		super();
	}

	public TabbyRequestException(String message, Throwable cause, boolean enableSuppression,
			boolean writableStackTrace) {
		super(message, cause, enableSuppression, writableStackTrace);
	}

	public TabbyRequestException(String message, Throwable cause) {
		super(message, cause);
	}

	public TabbyRequestException(String message) {
		super(message);
	}

	public TabbyRequestException(Throwable cause) {
		super(cause);
	}

}
