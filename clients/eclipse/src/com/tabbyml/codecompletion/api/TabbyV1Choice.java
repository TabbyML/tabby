package com.tabbyml.codecompletion.api;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public class TabbyV1Choice {
	private int index;
	private String text;
	
	public int getIndex() {
		return index;
	}



	public void setIndex(int index) {
		this.index = index;
	}



	public String getText() {
		return text;
	}



	public void setText(String text) {
		this.text = text;
	}



	@Override
	public String toString() {
		return text;
	}

}
