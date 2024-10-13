package com.tabbyml.tabby4eclipse;

import java.net.URL;

import org.eclipse.core.runtime.FileLocator;
import org.eclipse.core.runtime.Path;
import org.eclipse.core.runtime.Platform;
import org.eclipse.jface.resource.ImageDescriptor;
import org.eclipse.jface.resource.ImageRegistry;
import org.eclipse.swt.graphics.Image;
import org.osgi.framework.Bundle;

public class Images {
	private static Bundle bundle = Platform.getBundle(Activator.PLUGIN_ID);
	private static ImageRegistry icons = Activator.getDefault().getImageRegistry();

	public static final String ICON_CHAT = "chat.png";
	public static final String ICON_CHECK = "check_tsk.png";
	public static final String ICON_ERROR = "hprio_tsk.png";
	public static final String ICON_WARN = "warn_tsk.png";
	public static final String ICON_LOADING = "progress_task.png";
	public static final String ICON_SETTINGS = "settings.png";

	public static Image getIcon(String filename) {
		Image icon = icons.get(filename);
		if (icon == null) {
			icon = createImageFromFile(filename);
			icons.put(filename, icon);
		}
		return icon;
	}
	
	private static Image createImageFromFile(String filename) {
		String path = "images/" + filename;
		URL url = FileLocator.find(bundle, new Path(path));
		ImageDescriptor imageDesc = ImageDescriptor.createFromURL(url);
		return imageDesc.createImage();
	}
}
