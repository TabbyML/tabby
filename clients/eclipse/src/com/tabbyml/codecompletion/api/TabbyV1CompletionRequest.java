package com.tabbyml.codecompletion.api;

import java.util.HashMap;
import java.util.Map;

public class TabbyV1CompletionRequest {
	private String language;
	private Map<String, String> segments;
	
	public TabbyV1CompletionRequest(String prefix, String suffix) {
		super();
		this.language = "java";
		this.segments = new HashMap<String, String>();
		this.segments.put("prefix", prefix);
		this.segments.put("suffix", suffix);
	}
	
	public String getLanguage() {
		return language;
	}
	public void setLanguage(String language) {
		this.language = language;
	}
	public Map<String, String> getSegments() {
		return segments;
	}
	public void setSegments(Map<String, String> segments) {
		this.segments = segments;
	}	

}
