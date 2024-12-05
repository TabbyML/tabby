package com.tabbyml.tabby4eclipse.lsp.protocol;

public class StatusRequestParams {
	private boolean recheckConnection;

	public StatusRequestParams() {
	}

	public boolean getRecheckConnection() {
		return recheckConnection;
	}

	public void setRecheckConnection(boolean recheckConnection) {
		this.recheckConnection = recheckConnection;
	}
}
