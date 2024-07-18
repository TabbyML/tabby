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
			Bundle bundle = Platform.getBundle(Activator.PLUGIN_ID);
			URL agentScriptUrl = FileLocator.find(bundle, new Path("tabby-agent/dist/node/index.js"));
			if (agentScriptUrl == null) {
	            logger.error("Cannot find tabby-agent script.");
	            return;
	        }
			File agentScriptFile = new File(FileLocator.toFileURL(agentScriptUrl).getPath());
			List<String> commands = List.of("node", agentScriptFile.getAbsolutePath(), "--stdio");
			logger.info("Will use command " + commands.toString() + " to start Tabby language server.");
			this.setCommands(commands);
		} catch (IOException e) {
			logger.error("Failed to locate tabby-agent script.", e);
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
