package com.tabbyml.tabby4eclipse;

import org.eclipse.ui.plugin.AbstractUIPlugin;
import org.osgi.framework.BundleContext;

/**
 * The activator class controls the plug-in life cycle
 */
public class Activator extends AbstractUIPlugin {
	// The plug-in ID
	public static final String PLUGIN_ID = "com.tabbyml.tabby4eclipse";

	// The shared instance
	private static Activator plugin;

	public static Activator getDefault() {
		return plugin;
	}

	private Logger logger = new Logger("Activator");

	public Activator() {
	}

	@Override
	public void start(BundleContext context) throws Exception {
		super.start(context);
		plugin = this;
		logger.info("Tabby plugin is starting.");
	}

	@Override
	public void stop(BundleContext context) throws Exception {
		logger.info("Tabby plugin is stopping.");
		plugin = null;
		super.stop(context);
	}
}
