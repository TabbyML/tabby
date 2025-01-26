package com.tabbyml.tabby4eclipse;

public class Version {
	private int major;
	private int minor;
	private int patch;

	public Version(String versionStr) {
		int startIndex = 0;
		while (startIndex < versionStr.length() && !Character.isDigit(versionStr.charAt(startIndex))) {
			startIndex++;
		}
		if (startIndex >= versionStr.length()) {
			return;
		}
		int endIndex = versionStr.indexOf("-");
		String numPart = (endIndex != -1) ? versionStr.substring(startIndex, endIndex)
				: versionStr.substring(startIndex);

		String[] parts = numPart.split("\\.");
		if (parts.length > 0) {
			this.major = parseInt(parts[0]);
		}
		if (parts.length > 1) {
			this.minor = parseInt(parts[1]);
		}
		if (parts.length > 2) {
			this.patch = parseInt(parts[2]);
		}
	}

	public int getMajor() {
		return major;
	}

	public int getMinor() {
		return minor;
	}

	public int getPatch() {
		return patch;
	}

	public boolean isGreaterOrEqualThan(Version other) {
		if (this.major > other.major) {
			return true;
		} else if (this.major < other.major) {
			return false;
		} else {
			if (this.minor > other.minor) {
				return true;
			} else if (this.minor < other.minor) {
				return false;
			} else {
				return this.patch >= other.patch;
			}
		}
	}

	public boolean isEqual(Version other, boolean ignorePatch) {
		if (this.major != other.major || this.minor != other.minor) {
			return false;
		}
		if (ignorePatch) {
			return true;
		}
		return this.patch == other.patch;
	}

	public boolean isZero() {
		return this.major == 0 && this.minor == 0 && this.patch == 0;
	}

	private int parseInt(String str) {
		try {
			return Integer.parseInt(str);
		} catch (NumberFormatException e) {
			return 0;
		}
	}
}