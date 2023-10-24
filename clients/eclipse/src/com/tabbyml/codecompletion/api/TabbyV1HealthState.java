package com.tabbyml.codecompletion.api;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public class TabbyV1HealthState {
	
	private String model;
	private String chatModel;
	private String device;
	private String arch;
	private String cpuInfo;
	private int cpuCount;
	private List<String> cudaDevices;
	private TabbyV1Version version;
	public String getModel() {
		return model;
	}
	public void setModel(String model) {
		this.model = model;
	}
	public String getChatModel() {
		return chatModel;
	}
	public void setChatModel(String chatModel) {
		this.chatModel = chatModel;
	}
	public String getDevice() {
		return device;
	}
	public void setDevice(String device) {
		this.device = device;
	}
	public String getArch() {
		return arch;
	}
	public void setArch(String arch) {
		this.arch = arch;
	}
	public String getCpuInfo() {
		return cpuInfo;
	}
	public void setCpuInfo(String cpuInfo) {
		this.cpuInfo = cpuInfo;
	}
	public int getCpuCount() {
		return cpuCount;
	}
	public void setCpuCount(int cpuCount) {
		this.cpuCount = cpuCount;
	}
	public List<String> getCudaDevices() {
		return cudaDevices;
	}
	public void setCudaDevices(List<String> cudaDevices) {
		this.cudaDevices = cudaDevices;
	}
	public TabbyV1Version getVersion() {
		return version;
	}
	public void setVersion(TabbyV1Version version) {
		this.version = version;
	}

}
