package com.tabbyml.tabby4eclipse.lsp;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URL;
import java.util.List;

import org.eclipse.core.runtime.FileLocator;
import org.eclipse.core.runtime.Path;
import org.eclipse.core.runtime.Platform;
import org.eclipse.lsp4e.server.ProcessStreamConnectionProvider;
import org.osgi.framework.Bundle;

import com.tabbyml.tabby4eclipse.Activator;
import com.tabbyml.tabby4eclipse.Logger;
import com.tabbyml.tabby4eclipse.lsp.protocol.ClientCapabilities;
import com.tabbyml.tabby4eclipse.lsp.protocol.ClientCapabilities.TabbyClientCapabilities;
import com.tabbyml.tabby4eclipse.lsp.protocol.ClientCapabilities.TextDocumentClientCapabilities;
import com.tabbyml.tabby4eclipse.lsp.protocol.ClientInfo;
import com.tabbyml.tabby4eclipse.lsp.protocol.ClientInfo.TabbyPluginInfo;
import com.tabbyml.tabby4eclipse.lsp.protocol.ClientProvidedConfig;
import com.tabbyml.tabby4eclipse.lsp.protocol.InitializationOptions;

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
					File file = new File(p, isWindows() ? "node.exe" : "node");
					if (file.exists() && file.canExecute()) {
						nodeExecutableFile = file;
						logger.info("Node executable: " + file.getAbsolutePath());
						break;
					}
				}
			}
			if (nodeExecutableFile == null) {
				StatusInfoHolder.getInstance().setConnectionFailed(true);
				logger.error("Cannot find node executable.");
				return;
			}
			// Find tabby-agent script
			Bundle bundle = Platform.getBundle(Activator.PLUGIN_ID);
			URL agentScriptUrl = FileLocator.find(bundle, new Path("tabby-agent/dist/node/index.js"));
			if (agentScriptUrl == null) {
				StatusInfoHolder.getInstance().setConnectionFailed(true);
				logger.error("Cannot find tabby-agent script.");
				return;
			}
			File agentScriptFile = new File(FileLocator.toFileURL(agentScriptUrl).getPath());
			// Setup command to start tabby-agent
			List<String> commands = List.of(nodeExecutableFile.getAbsolutePath(), agentScriptFile.getAbsolutePath(),
					"--stdio");
			logger.info("Will use command " + commands.toString() + " to start Tabby language server.");
			this.setCommands(commands);
		} catch (IOException e) {
			StatusInfoHolder.getInstance().setConnectionFailed(true);
			logger.error("Failed to setup command to start Tabby language server.", e);
		}
	}

	private static boolean isWindows() {
		return System.getProperty("os.name").toLowerCase().contains("win");
	}

	@Override
	public Object getInitializationOptions(URI rootUri) {
		return new InitializationOptions(getProvidedConfig(), getClientInfo(), getClientCapabilities());
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

	private ClientProvidedConfig getProvidedConfig() {
		ClientProvidedConfig config = new ClientProvidedConfig();
		return config;
	}

	private ClientInfo getClientInfo() {
		TabbyPluginInfo tabbyPluginInfo = new TabbyPluginInfo();
		Bundle bundle = Platform.getBundle(Activator.PLUGIN_ID);
		tabbyPluginInfo.setName(Activator.PLUGIN_ID);
		tabbyPluginInfo.setVersion(bundle.getVersion().toString());

		ClientInfo clientInfo = new ClientInfo();
		clientInfo.setTabbyPlugin(tabbyPluginInfo);
		return clientInfo;
	}

	private ClientCapabilities getClientCapabilities() {
		TextDocumentClientCapabilities textDocumentClientCapabilities = new TextDocumentClientCapabilities();
		textDocumentClientCapabilities.setCompletion(false);
		textDocumentClientCapabilities.setInlineCompletion(true);

		TabbyClientCapabilities tabbyClientCapabilities = new TabbyClientCapabilities();
		tabbyClientCapabilities.setConfigDidChangeListener(true);
		tabbyClientCapabilities.setStatusDidChangeListener(true);
		tabbyClientCapabilities.setWorkspaceFileSystem(true);
		tabbyClientCapabilities.setGitProvider(true);
		tabbyClientCapabilities.setLanguageSupport(true);

		ClientCapabilities clientCapabilities = new ClientCapabilities();
		clientCapabilities.setTextDocument(textDocumentClientCapabilities);
		clientCapabilities.setTabby(tabbyClientCapabilities);
		return clientCapabilities;
	}
}
