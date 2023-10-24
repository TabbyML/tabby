package com.tabbyml.codecompletion.api;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public class TabbyV1Version {
	private String buildDate;
	private String buildTimestamp;
	private String gitSha;
	private String gitDescribe;
	public String getBuildDate() {
		return buildDate;
	}
	public void setBuildDate(String buildDate) {
		this.buildDate = buildDate;
	}
	public String getBuildTimestamp() {
		return buildTimestamp;
	}
	public void setBuildTimestamp(String buildTimestamp) {
		this.buildTimestamp = buildTimestamp;
	}
	public String getGitSha() {
		return gitSha;
	}
	public void setGitSha(String gitSha) {
		this.gitSha = gitSha;
	}
	public String getGitDescribe() {
		return gitDescribe;
	}
	public void setGitDescribe(String gitDescribe) {
		this.gitDescribe = gitDescribe;
	}

}
