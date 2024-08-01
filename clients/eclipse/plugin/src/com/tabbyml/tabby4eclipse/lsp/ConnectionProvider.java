package com.tabbyml.tabby4eclipse.lsp;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.List;

import org.eclipse.core.runtime.FileLocator;
import org.eclipse.core.runtime.Path;
import org.eclipse.core.runtime.Platform;
import org.eclipse.lsp4e.server.ProcessStreamConnectionProvider;
import org.osgi.framework.Bundle;

import com.tabbyml.tabby4eclipse.Activator;
import com.tabbyml.tabby4eclipse.Logger;

public class ConnectionProvider extends ProcessStreamConnectionProvider {
	private Logger logger = new Logger("ConnectionProvider");

	public ConnectionProvider() {
		try {
			// Find node executable
			File nodeExecutableFile = null;
			String systemPath = System.getenv("PATH");
			logger.info("System env PATH: " + systemPath);
			if (systemPath != null) {
				String[] paths = systemPath.split(File.pathSeparator);
				for (String p : paths) {
					File file = new File(p, "node");
					if (file.exists() && file.canExecute()) {
						nodeExecutableFile = file;
						logger.info("Node executable: " + file.getAbsolutePath());
						break;
					}
				}
			}
			if (nodeExecutableFile == null) {
				logger.error("Cannot find node executable.");
				return;
			}
			// Find tabby-agent script
			Bundle bundle = Platform.getBundle(Activator.PLUGIN_ID);
			URL agentScriptUrl = FileLocator.find(bundle, new Path("tabby-agent/dist/node/index.js"));
			if (agentScriptUrl == null) {
				logger.error("Cannot find tabby-agent script.");
				return;
			}
			File agentScriptFile = new File(FileLocator.toFileURL(agentScriptUrl).getPath());
			// Setup command to start tabby-agent
			List<String> commands = List.of(nodeExecutableFile.getAbsolutePath(), agentScriptFile.getAbsolutePath(), "--stdio");
			logger.info("Will use command " + commands.toString() + " to start Tabby language server.");
			this.setCommands(commands);
		} catch (IOException e) {
			logger.error("Failed to setup command to start Tabby language server.", e);
		}
	}

	@Override
	public void start() throws IOException {
		super.start();
		logger.info("Tabby language server started.");
	}

	@Override
	public void stop() {
		super.stop();
		logger.info("Tabby language server stopped.");
	}
}
