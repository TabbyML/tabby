package com.tabbyml.codecompletion.api;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public class TabbyV1CompletionResponse {
	@SuppressWarnings("unused")
	private String id;
	private List<TabbyV1Choice> choices;
	
	
	
	public String getId() {
		return id;
	}



	public void setId(String id) {
		this.id = id;
	}



	public List<TabbyV1Choice> getChoices() {
		return choices;
	}



	public void setChoices(List<TabbyV1Choice> choices) {
		this.choices = choices;
	}



	@Override
	public String toString() {
		return choices.get(0).toString();
	}
}
