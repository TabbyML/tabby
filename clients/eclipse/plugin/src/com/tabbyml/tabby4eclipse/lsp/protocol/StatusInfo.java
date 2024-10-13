package com.tabbyml.tabby4eclipse.lsp.protocol;

import java.util.Map;

import org.eclipse.lsp4j.Command;

public class StatusInfo {
	private String status;
	private String tooltip;
	private Map<String, Object> serverHealth;
	private Command command;

	public StatusInfo() {
		this.status = Status.DISCONNECTED;
	}

	public String getStatus() {
		return status;
	}

	public void setStatus(String status) {
		this.status = status;
	}

	public String getTooltip() {
		return tooltip;
	}

	public void setTooltip(String tooltip) {
		this.tooltip = tooltip;
	}

	public Map<String, Object> getServerHealth() {
		return serverHealth;
	}

	public void setServerHealth(Map<String, Object> serverHealth) {
		this.serverHealth = serverHealth;
	}

	public Command getCommand() {
		return command;
	}

	public void setCommand(Command command) {
		this.command = command;
	}

	public static class Status {
		public static final String CONNECTING = "connecting";
		public static final String UNAUTHORIZED = "unauthorized";
		public static final String DISCONNECTED = "disconnected";
		public static final String READY = "ready";
		public static final String READY_FOR_AUTO_TRIGGER = "readyForAutoTrigger";
		public static final String READY_FOR_MANUAL_TRIGGER = "readyForManualTrigger";
		public static final String FETCHING = "fetching";
		public static final String COMPLETION_RESPONSE_SLOW = "completionResponseSlow";
	}
}
