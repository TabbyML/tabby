package com.tabbyml.tabby4eclipse.lsp;

import java.util.ArrayList;
import java.util.List;

import com.tabbyml.tabby4eclipse.lsp.protocol.StatusInfo;

public class StatusInfoHolder {
	public static StatusInfoHolder getInstance() {
		return LazyHolder.INSTANCE;
	}

	private static class LazyHolder {
		private static final StatusInfoHolder INSTANCE = new StatusInfoHolder();
	}

	private boolean isConnectionFailed = false;
	private StatusInfo statusInfo = new StatusInfo();
	private List<StatusDidChangeListener> listeners = new ArrayList<>();

	public boolean isConnectionFailed() {
		return isConnectionFailed;
	}

	public void setConnectionFailed(boolean isConnectionFailed) {
		if (this.isConnectionFailed == isConnectionFailed) {
			return;
		}
		this.isConnectionFailed = isConnectionFailed;
		for (StatusDidChangeListener listener : listeners) {
			listener.statusDidChange();
		}
	}

	public StatusInfo getStatusInfo() {
		return statusInfo;
	}

	public void setStatusInfo(StatusInfo statusInfo) {
		this.statusInfo = statusInfo;
		for (StatusDidChangeListener listener : listeners) {
			listener.statusDidChange();
		}
	}

	public void addStatusDidChangeListener(StatusDidChangeListener listener) {
		listeners.add(listener);
	}

	public interface StatusDidChangeListener {
		void statusDidChange();
	}
}
