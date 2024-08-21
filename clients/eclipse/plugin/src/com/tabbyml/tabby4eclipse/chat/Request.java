package com.tabbyml.tabby4eclipse.chat;

import java.util.List;

public class Request {
	private String method;
	private List<Object> params;

	public Request(String method, List<Object> params) {
		this.method = method;
		this.params = params;
	}

	public String getMethod() {
		return method;
	}

	public void setMethod(String method) {
		this.method = method;
	}

	public List<Object> getParams() {
		return params;
	}

	public void setParams(List<Object> params) {
		this.params = params;
	}
}
