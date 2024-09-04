package com.tabbyml.tabby4eclipse;

import org.eclipse.core.runtime.IStatus;
import org.eclipse.core.runtime.Status;

public class Logger {
	private static String TABBY4ECLIPSE_LOG_DEBUG = System.getenv("TABBY4ECLIPSE_LOG_DEBUG");
	private static boolean LOG_DEBUG = TABBY4ECLIPSE_LOG_DEBUG != null && !TABBY4ECLIPSE_LOG_DEBUG.isEmpty();

	private String tag;

	public Logger(String tag) {
		this.tag = tag;
	}

	public void trace(String message, Object obj) {
		System.out.println(tagString(message));
		System.out.println(obj);
	}

	public void debug(String message) {
		System.out.println(tagString(message));
		if (LOG_DEBUG) {
			logStatus(IStatus.INFO, String.format("[DEBUG] %s", message), null);
		}
	}

	public void info(String message) {
		logStatus(IStatus.INFO, message, null);
	}

	public void warn(String message, Throwable throwable) {
		logStatus(IStatus.WARNING, message, throwable);
	}

	public void error(String message) {
		logStatus(IStatus.ERROR, message, null);
	}

	public void error(String message, Throwable throwable) {
		logStatus(IStatus.ERROR, message, throwable);
	}

	private void logStatus(int severity, String message, Throwable throwable) {
		Status status = new Status(severity, Activator.PLUGIN_ID, tagString(message), throwable);
		Activator.getDefault().getLog().log(status);
	}

	private String tagString(String message) {
		return "[" + tag + "] " + message;
	}
}
