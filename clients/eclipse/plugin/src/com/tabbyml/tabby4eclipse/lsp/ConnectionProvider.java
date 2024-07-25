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
		List<String> commands = List.of("npx", "tabby-agent@1.7.0", "--stdio");
		logger.info("Will use command " + commands.toString() + " to start Tabby language server.");
		this.setCommands(commands);
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
