package com.tabbyml.tabby4eclipse;

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
		return String.format("%.0f, %.0f%%, %.0f%%", h, s, l);
	}
}
