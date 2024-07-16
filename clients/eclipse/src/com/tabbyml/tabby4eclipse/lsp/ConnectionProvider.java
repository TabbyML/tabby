package com.tabbyml.tabby4eclipse.lsp;

import java.io.IOException;
import java.util.List;

import org.eclipse.lsp4e.server.ProcessStreamConnectionProvider;
import com.tabbyml.tabby4eclipse.Logger;

public class ConnectionProvider extends ProcessStreamConnectionProvider {
	private Logger logger = new Logger("ConnectionProvider");
	
	public ConnectionProvider() {
		List<String> commands = List.of("npx", "tabby-agent", "--stdio");
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
