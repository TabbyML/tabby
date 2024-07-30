package com.tabbyml.tabby4eclipse;

import java.net.URL;

import org.eclipse.core.runtime.FileLocator;
import org.eclipse.core.runtime.Path;
import org.eclipse.core.runtime.Platform;
import org.eclipse.jface.resource.ImageDescriptor;
import org.eclipse.swt.graphics.Image;
import org.eclipse.swt.graphics.RGB;
import org.eclipse.ui.PlatformUI;
import org.osgi.framework.Bundle;

public class Images {
	private static Bundle bundle = Platform.getBundle(Activator.PLUGIN_ID);
	private static Logger logger = new Logger("Images");
	
	private static boolean isDark = isDarkTheme();
	private static Image iconCheck = null;

	public static Image getIconCheck() {
		if (iconCheck == null) {
			iconCheck = createImage(isDark ? "images/check-dark.png" : "images/check-light.png");
		}
		return iconCheck;
	}

	private static boolean isDarkTheme() {
		RGB bgColor = PlatformUI.getWorkbench().getThemeManager().getCurrentTheme().getColorRegistry()
				.getRGB("org.eclipse.ui.workbench.ACTIVE_TAB_BG_START");
		if (bgColor != null) {
			boolean isBgDark =  (bgColor.red + bgColor.green + bgColor.blue) / 3 < 128;
			logger.info("Detected theme: " + (isBgDark ? "dark" : "light"));
			return isBgDark;
		}
		logger.info("Cannot detect theme. Assuming light.");
		return false;
	}

	private static Image createImage(String path) {
		URL url = FileLocator.find(bundle, new Path(path));
		ImageDescriptor imageDesc = ImageDescriptor.createFromURL(url);
		return imageDesc.createImage();
	}
}
