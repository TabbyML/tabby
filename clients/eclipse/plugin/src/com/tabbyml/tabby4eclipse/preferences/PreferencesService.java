package com.tabbyml.tabby4eclipse.preferences;

import org.eclipse.jface.preference.IPreferenceStore;
import org.eclipse.jface.util.IPropertyChangeListener;
import org.eclipse.jface.util.PropertyChangeEvent;
import org.eclipse.lsp4j.DidChangeConfigurationParams;
import org.eclipse.lsp4j.services.WorkspaceService;

import com.tabbyml.tabby4eclipse.Activator;
import com.tabbyml.tabby4eclipse.Logger;
import com.tabbyml.tabby4eclipse.lsp.LanguageServerService;
import com.tabbyml.tabby4eclipse.lsp.protocol.ClientProvidedConfig;
import com.tabbyml.tabby4eclipse.lsp.protocol.ILanguageServer;

public class PreferencesService {
	public static final String KEY_SERVER_ENDPOINT = "SERVER_ENDPOINT";
	public static final String KEY_SERVER_TOKEN = "SERVER_TOKEN";
	public static final String KEY_INLINE_COMPLETION_TRIGGER_AUTO = "INLINE_COMPLETION_TRIGGER_AUTO";
	public static final String KEY_NODE_BINARY_PATH = "NODE_BINARY_PATH";
	public static final String KEY_ANONYMOUS_USAGE_TRACKING_DISABLED = "ANONYMOUS_USAGE_TRACKING_DISABLED";

	public static PreferencesService getInstance() {
		return LazyHolder.INSTANCE;
	}

	private static class LazyHolder {
		private static final PreferencesService INSTANCE = new PreferencesService();
	}

	private Logger logger = new Logger("PreferencesService");

	public void init() {
		getPreferenceStore().setDefault(KEY_INLINE_COMPLETION_TRIGGER_AUTO, true);

		getPreferenceStore().addPropertyChangeListener(new IPropertyChangeListener() {
			@Override
			public void propertyChange(PropertyChangeEvent event) {
				logger.info("Syncing configuration.");
				LanguageServerService.getInstance().getServer().execute((server) -> {
					WorkspaceService workspaceService = ((ILanguageServer) server).getWorkspaceService();
					DidChangeConfigurationParams params = new DidChangeConfigurationParams();
					params.setSettings(buildClientProvidedConfig());
					workspaceService.didChangeConfiguration(params);
					return null;
				});
			}
		});
	}

	public IPreferenceStore getPreferenceStore() {
		return Activator.getDefault().getPreferenceStore();
	}

	public ClientProvidedConfig buildClientProvidedConfig() {
		ClientProvidedConfig config = new ClientProvidedConfig();

		ClientProvidedConfig.ServerConfig serverConfig = new ClientProvidedConfig.ServerConfig(getServerEndpoint(),
				getServerToken());
		config.setServer(serverConfig);

		ClientProvidedConfig.InlineCompletionConfig inlineCompletionConfig = new ClientProvidedConfig.InlineCompletionConfig(
				getInlineCompletionTriggerAuto() ? ClientProvidedConfig.InlineCompletionConfig.TriggerMode.AUTO
						: ClientProvidedConfig.InlineCompletionConfig.TriggerMode.MANUAL);
		config.setInlineCompletion(inlineCompletionConfig);

		ClientProvidedConfig.AnonymousUsageTrackingConfig anonymousUsageTrackingConfig = new ClientProvidedConfig.AnonymousUsageTrackingConfig(
				getAnonymousUsageTrackingDisabled());
		config.setAnonymousUsageTracking(anonymousUsageTrackingConfig);

		return config;
	}

	public String getServerEndpoint() {
		return getPreferenceStore().getString(KEY_SERVER_ENDPOINT);
	}

	public String getServerToken() {
		return getPreferenceStore().getString(KEY_SERVER_TOKEN);
	}

	public boolean getInlineCompletionTriggerAuto() {
		return getPreferenceStore().getBoolean(KEY_INLINE_COMPLETION_TRIGGER_AUTO);
	}

	public String getNodeBinaryPath() {
		return getPreferenceStore().getString(KEY_NODE_BINARY_PATH);
	}

	public boolean getAnonymousUsageTrackingDisabled() {
		return getPreferenceStore().getBoolean(KEY_ANONYMOUS_USAGE_TRACKING_DISABLED);
	}
}
