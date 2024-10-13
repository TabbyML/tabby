package com.tabbyml.tabby4eclipse.chat;

import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.eclipse.core.runtime.FileLocator;
import org.eclipse.core.runtime.Path;
import org.eclipse.core.runtime.Platform;
import org.eclipse.jface.resource.ColorRegistry;
import org.eclipse.jface.resource.FontRegistry;
import org.eclipse.swt.SWT;
import org.eclipse.swt.browser.Browser;
import org.eclipse.swt.browser.BrowserFunction;
import org.eclipse.swt.browser.ProgressAdapter;
import org.eclipse.swt.browser.ProgressEvent;
import org.eclipse.swt.graphics.Color;
import org.eclipse.swt.graphics.FontData;
import org.eclipse.swt.graphics.RGB;
import org.eclipse.swt.layout.FillLayout;
import org.eclipse.swt.widgets.Composite;
import org.eclipse.ui.PlatformUI;
import org.eclipse.ui.part.ViewPart;
import org.eclipse.ui.themes.ITheme;
import org.osgi.framework.Bundle;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.tabbyml.tabby4eclipse.Activator;
import com.tabbyml.tabby4eclipse.Logger;
import com.tabbyml.tabby4eclipse.StringUtils;
import com.tabbyml.tabby4eclipse.Utils;
import com.tabbyml.tabby4eclipse.chat.ChatMessage.FileContext;
import com.tabbyml.tabby4eclipse.lsp.LanguageServerService;
import com.tabbyml.tabby4eclipse.lsp.ServerConfigHolder;
import com.tabbyml.tabby4eclipse.lsp.StatusInfoHolder;
import com.tabbyml.tabby4eclipse.lsp.protocol.Config;
import com.tabbyml.tabby4eclipse.lsp.protocol.ILanguageServer;
import com.tabbyml.tabby4eclipse.lsp.protocol.IStatusService;
import com.tabbyml.tabby4eclipse.lsp.protocol.StatusInfo;
import com.tabbyml.tabby4eclipse.lsp.protocol.StatusRequestParams;

public class ChatView extends ViewPart {
	private Logger logger = new Logger("ChatView");
	private Gson gson = new Gson();

	private StatusInfoHolder statusInfoHolder = StatusInfoHolder.getInstance();
	private ServerConfigHolder serverConfigHolder = ServerConfigHolder.getInstance();

	private Browser browser;
	private List<BrowserFunction> browserFunctions = new ArrayList<>();

	private boolean isHtmlLoaded = false;
	private boolean isChatPanelLoaded = false;
	private List<String> pendingScripts = new ArrayList<>();
	private Config.ServerConfig currentConfig;

	private boolean isDark;
	private RGB bgColor;
	private RGB bgActiveColor;
	private RGB fgColor;
	private RGB borderColor;
	private RGB primaryColor;
	private String font;
	private int fontSize = 13;

	@Override
	public void createPartControl(Composite parent) {
		setupThemeStyle();
		parent.setLayout(new FillLayout());

		browser = new Browser(parent, Utils.isWindows() ? SWT.EDGE : SWT.WEBKIT);
		browser.setBackground(new Color(bgActiveColor));
		browser.setVisible(false);

		browser.addProgressListener(new ProgressAdapter() {
			@Override
			public void completed(ProgressEvent event) {
				handleLoaded();
			}
		});
		// Inject callbacks
		browserFunctions.add(new BrowserFunction(browser, "handleReload") {
			@Override
			public Object function(Object[] arguments) {
				reloadContent(true);
				return null;
			}
		});

		browserFunctions.add(new BrowserFunction(browser, "handleChatPanelRequest") {
			@Override
			public Object function(Object[] arguments) {
				if (arguments.length > 0) {
					logger.info("HandleChatPanelRequest: " + arguments[0]);
					Request request = gson.fromJson(arguments[0].toString(), Request.class);
					handleChatPanelRequest(request);
				}
				return null;
			}
		});

		load();
		serverConfigHolder.addConfigDidChangeListener(() -> {
			reloadContent(false);
		});
		statusInfoHolder.addStatusDidChangeListener(() -> {
			reloadContent(false);
		});
	}

	@Override
	public void setFocus() {
		browser.forceFocus();
	}

	@Override
	public void dispose() {
		if (browser != null && !browser.isDisposed()) {
			browser.dispose();
		}
		if (!browserFunctions.isEmpty()) {
			browserFunctions.forEach(f -> f.dispose());
			browserFunctions.clear();
		}
		super.dispose();
	}

	public void explainSelectedText() {
		sendRequestToChatPanel(new Request("sendMessage", new ArrayList<>() {
			{
				ChatMessage chatMessage = new ChatMessage();
				chatMessage.setMessage(ChatViewUtils.PROMPT_EXPLAIN);
				chatMessage.setSelectContext(ChatViewUtils.getSelectedTextAsFileContext());
				add(chatMessage);
			}
		}));
	}

	public void fixSelectedText() {
		// FIXME(@icycodes): collect the diagnostic message provided by IDE or LSP
		sendRequestToChatPanel(new Request("sendMessage", new ArrayList<>() {
			{
				ChatMessage chatMessage = new ChatMessage();
				chatMessage.setMessage(ChatViewUtils.PROMPT_FIX);
				chatMessage.setSelectContext(ChatViewUtils.getSelectedTextAsFileContext());
				add(chatMessage);
			}
		}));
	}

	public void generateDocsForSelectedText() {
		sendRequestToChatPanel(new Request("sendMessage", new ArrayList<>() {
			{
				ChatMessage chatMessage = new ChatMessage();
				chatMessage.setMessage(ChatViewUtils.PROMPT_GENERATE_DOCS);
				chatMessage.setSelectContext(ChatViewUtils.getSelectedTextAsFileContext());
				add(chatMessage);
			}
		}));
	}

	public void generateTestsForSelectedText() {
		sendRequestToChatPanel(new Request("sendMessage", new ArrayList<>() {
			{
				ChatMessage chatMessage = new ChatMessage();
				chatMessage.setMessage(ChatViewUtils.PROMPT_GENERATE_TESTS);
				chatMessage.setSelectContext(ChatViewUtils.getSelectedTextAsFileContext());
				add(chatMessage);
			}
		}));
	}

	public void addSelectedTextAsContext() {
		sendRequestToChatPanel(new Request("addRelevantContext", new ArrayList<>() {
			{
				add(ChatViewUtils.getSelectedTextAsFileContext());
			}
		}));
	}

	public void addActiveEditorAsContext() {
		sendRequestToChatPanel(new Request("addRelevantContext", new ArrayList<>() {
			{
				add(ChatViewUtils.getActiveEditorAsFileContext());
			}
		}));
	}

	private void setupThemeStyle() {
		ITheme currentTheme = PlatformUI.getWorkbench().getThemeManager().getCurrentTheme();
		ColorRegistry colorRegistry = currentTheme.getColorRegistry();
		bgColor = colorRegistry.getRGB("org.eclipse.ui.workbench.ACTIVE_TAB_BG_START");
		bgActiveColor = colorRegistry.getRGB("org.eclipse.ui.workbench.ACTIVE_TAB_BG_END");
		fgColor = colorRegistry.getRGB("org.eclipse.ui.workbench.ACTIVE_TAB_TEXT_COLOR");
		borderColor = colorRegistry.getRGB("org.eclipse.ui.workbench.ACTIVE_TAB_INNER_KEYLINE_COLOR");
		primaryColor = colorRegistry.getRGB("org.eclipse.ui.workbench.LINK_COLOR");
		isDark = (bgColor.red + bgColor.green + bgColor.blue) / 3 < 128;

		FontRegistry fontRegistry = currentTheme.getFontRegistry();
		FontData[] fontData = fontRegistry.getFontData("org.eclipse.jface.textfont");
		if (fontData.length > 0) {
			font = fontData[0].getName();
			fontSize = fontData[0].getHeight();
		}
	}

	private String buildCss() {
		String css = "";
		if (bgActiveColor != null) {
			css += String.format("background-color: hsl(%s);", StringUtils.toHsl(bgActiveColor));
		}
		if (bgColor != null) {
			css += String.format("--background: %s;", StringUtils.toHsl(bgColor));
		}
		if (fgColor != null) {
			css += String.format("--foreground: %s;", StringUtils.toHsl(fgColor));
		}
		if (borderColor != null) {
			css += String.format("--border: %s;", StringUtils.toHsl(borderColor));
		}
		if (primaryColor != null) {
			css += String.format("--primary: %s;", StringUtils.toHsl(primaryColor));
		}
		if (font != null) {
			css += String.format("font: %s;", font);
		}
		css += String.format("font-size: %spt;", fontSize);
		return css;
	}

	private void load() {
		try {
			// Find chat panel html file
			Bundle bundle = Platform.getBundle(Activator.PLUGIN_ID);
			URL chatPanelPath = FileLocator.find(bundle, new Path("chat-panel/index.html"));
			if (chatPanelPath == null) {
				logger.error("Failed to find chat panel html file.");
				return;
			}
			URL url = FileLocator.toFileURL(chatPanelPath);
			browser.getDisplay().asyncExec(() -> {
				logger.info("Load url: " + url.toString());
				browser.setUrl(url.toString());
			});
		} catch (Exception e) {
			logger.error("Failed to load chat panel html file.", e);
		}
	}

	private void handleLoaded() {
		isHtmlLoaded = true;
		isChatPanelLoaded = false;
		applyStyle();
		reloadContent(false);
	}

	private void reloadContent(boolean force) {
		if (!isHtmlLoaded) {
			return;
		}
		if (force) {
			LanguageServerService.getInstance().getServer().execute((server) -> {
				IStatusService statusService = ((ILanguageServer) server).getStatusService();
				StatusRequestParams params = new StatusRequestParams();
				params.setRecheckConnection(true);
				return statusService.getStatus(params);
			}).thenAccept((statusInfo) -> {
				String status = statusInfo.getStatus();
				reloadContentForStatus(status, true);
			});
		} else {
			String status = statusInfoHolder.getStatusInfo().getStatus();
			reloadContentForStatus(status, false);
		}
	}

	private void reloadContentForStatus(String status, boolean force) {
		if (status.equals(StatusInfo.Status.DISCONNECTED)) {
			updateContentToMessage("Cannot connect to Tabby server, please check your settings.");
		} else if (status.equals(StatusInfo.Status.CONNECTING)) {
			updateContentToMessage("Connecting to Tabby server...");
		} else if (status.equals(StatusInfo.Status.UNAUTHORIZED)) {
			updateContentToMessage("Authorization required, please set your token in settings.");
		} else {
			Map<String, Object> serverHealth = statusInfoHolder.getStatusInfo().getServerHealth();
			String error = ChatViewUtils.checkServerHealth(serverHealth);
			if (error != null) {
				updateContentToMessage(error);
			} else {
				// Load main
				Config.ServerConfig config = serverConfigHolder.getConfig().getServer();
				if (config == null) {
					updateContentToMessage("Initializing...");
				} else if (force || currentConfig == null || currentConfig.getEndpoint() != config.getEndpoint()
						|| currentConfig.getToken() != config.getToken()) {
					updateContentToMessage("Loading chat panel...");
					isChatPanelLoaded = false;
					currentConfig = config;
					loadChatPanel();
				}
			}
		}
	}

	private void updateContentToMessage(String message) {
		showMessage(message);
		showChatPanel(false);
	}

	private void updateContentToChatPanel() {
		showMessage(null);
		showChatPanel(true);
	}

	private void showMessage(String message) {
		if (message != null) {
			executeScript(String.format("showMessage('%s')", message));
		} else {
			executeScript("showMessage(undefined)");
		}
	}

	private void showChatPanel(boolean visible) {
		executeScript(String.format("showChatPanel(%s)", visible ? "true" : "false"));
	}

	private void loadChatPanel() {
		String chatUrl = String.format("%s/chat?client=eclipse", currentConfig.getEndpoint());
		executeScript(String.format("loadChatPanel('%s')", chatUrl));
	}

	private void applyStyle() {
		String theme = isDark ? "dark" : "light";
		String css = buildCss();
		String json = gson.toJson(new HashMap<>() {
			{
				put("theme", theme);
				put("css", css);
			}
		});
		executeScript(String.format("applyStyle('%s')", json));
		browser.setVisible(true);
	}

	private void sendRequestToChatPanel(Request request) {
		String json = gson.toJson(request);
		String script = String.format("sendRequestToChatPanel('%s')", StringUtils.escapeCharacters(json));
		if (isChatPanelLoaded) {
			executeScript(script);
		} else {
			pendingScripts.add(script);
		}
	}

	private void executeScript(String script) {
		browser.getDisplay().asyncExec(() -> {
			browser.execute(script);
		});
	}

	private void handleChatPanelRequest(Request request) {
		switch (request.getMethod()) {
		case "navigate": {
			List<Object> params = request.getParams();
			if (params.size() < 1) {
				return;
			}
			FileContext context = gson.fromJson(gson.toJson(params.get(0)), FileContext.class);
			ChatViewUtils.navigateToFileContext(context);
			break;
		}
		case "refresh": {
			reloadContent(true);
			break;
		}
		case "onSubmitMessage": {
			List<Object> params = request.getParams();
			if (params.size() < 1) {
				return;
			}
			String message = (String) params.get(0);
			List<FileContext> relevantContexts = params.size() > 1
					? relevantContexts = gson.fromJson(gson.toJson(params.get(1)), new TypeToken<List<FileContext>>() {
					}.getType())
					: null;
			sendRequestToChatPanel(new Request("sendMessage", new ArrayList<>() {
				{
					ChatMessage chatMessage = new ChatMessage();
					chatMessage.setMessage(message);
					chatMessage.setRelevantContext(relevantContexts);
					chatMessage.setActiveContext(ChatViewUtils.getSelectedTextAsFileContext());
					add(chatMessage);
				}
			}));
			break;
		}
		case "onApplyInEditor": {
			List<Object> params = request.getParams();
			if (params.size() < 1) {
				return;
			}
			String content = (String) params.get(0);
			ChatViewUtils.applyContentInEditor(content);
			break;
		}
		case "onLoaded": {
			List<Object> params = request.getParams();
			if (params.size() < 1) {
				return;
			}
			Map<String, Object> onLoadedParams = (Map<String, Object>) params.get(0);
			String apiVersion = (String) onLoadedParams.getOrDefault("apiVersion", "");
			if (!apiVersion.isBlank()) {
				String error = ChatViewUtils.checkChatPanelApiVersion(apiVersion);
				if (error != null) {
					updateContentToMessage(error);
					return;
				}
			}
			initChatPanel();
			break;
		}
		case "onCopy": {
			List<Object> params = request.getParams();
			if (params.size() < 1) {
				return;
			}
			String content = (String) params.get(0);
			ChatViewUtils.setClipboardContent(content);
			break;
		}
		case "onKeyboardEvent": {
			// FIXME: For macOS and windows, the eclipse keyboard shortcuts are not
			// available when browser is focused,
			// we should handle keyboard events here.
			break;
		}
		}
	}

	private void initChatPanel() {
		isChatPanelLoaded = true;
		sendRequestToChatPanel(new Request("init", new ArrayList<>() {
			{
				add(new HashMap<>() {
					{
						put("fetcherOptions", new HashMap<>() {
							{
								put("authorization", currentConfig.getToken());
							}
						});
					}
				});
			}
		}));
		sendRequestToChatPanel(new Request("updateTheme", new ArrayList<>() {
			{
				add(buildCss());
				add(isDark ? "dark" : "light");
			}
		}));
		browser.getDisplay().timerExec(100, () -> {
			updateContentToChatPanel();
			pendingScripts.forEach((script) -> {
				executeScript(script);
			});
			pendingScripts.clear();
		});
	}

}
