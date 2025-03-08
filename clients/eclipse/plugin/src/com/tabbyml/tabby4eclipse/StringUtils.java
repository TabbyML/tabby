package com.tabbyml.tabby4eclipse;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.eclipse.swt.graphics.RGB;

public class StringUtils {
	public static String escapeCharacters(String jsonString) {
		return jsonString.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r")
				.replace("\t", "\\t");
	}

	public static String toHsl(RGB rgb) {
		double r = rgb.red / 255.0;
		double g = rgb.green / 255.0;
		double b = rgb.blue / 255.0;
		double max = Math.max(r, Math.max(g, b));
		double min = Math.min(r, Math.min(g, b));
		double l = (max + min) / 2.0;
		double h, s;
		if (max == min) {
			h = 0;
			s = 0;
		} else {
			double delta = max - min;
			s = l > 0.5 ? delta / (2.0 - max - min) : delta / (max + min);
			if (max == r) {
				h = (g - b) / delta + (g < b ? 6 : 0);
			} else if (max == g) {
				h = (b - r) / delta + 2;
			} else {
				h = (r - g) / delta + 4;
			}
			h /= 6;
		}
		h *= 360;
		s *= 100;
		l *= 100;
		return String.format("%.0f %.0f%% %.0f%%", h, s, l);
	}

	public static class TextWithTabs {
		private int tabs;
		private String text;

		public TextWithTabs(int tabs, String text) {
			this.tabs = tabs;
			this.text = text;
		}

		public int getTabs() {
			return tabs;
		}

		public String getText() {
			return text;
		}
	}

	static final Pattern PATTERN_LEADING_TABS = Pattern.compile("^(\\t*)(.*)$");

	public static TextWithTabs splitLeadingTabs(String text) {
		Matcher matcher = PATTERN_LEADING_TABS.matcher(text);
		if (matcher.matches()) {
			return new TextWithTabs(matcher.group(1).length(), matcher.group(2));
		} else {
			return new TextWithTabs(0, text);
		}
	}
}
