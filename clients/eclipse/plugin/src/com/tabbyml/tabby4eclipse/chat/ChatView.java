package com.tabbyml.tabby4eclipse.chat;

import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

import org.eclipse.core.runtime.FileLocator;
import org.eclipse.core.runtime.Path;
import org.eclipse.core.runtime.Platform;
import org.eclipse.jface.resource.ColorRegistry;
import org.eclipse.jface.resource.FontRegistry;
import org.eclipse.jface.text.ITextSelection;
import org.eclipse.jface.viewers.ISelection;
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
import org.eclipse.ui.ISelectionListener;
import org.eclipse.ui.IWorkbenchPart;
import org.eclipse.ui.PlatformUI;
import org.eclipse.ui.part.ViewPart;
import org.eclipse.ui.themes.ITheme;
import org.osgi.framework.Bundle;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.tabbyml.tabby4eclipse.Activator;
import com.tabbyml.tabby4eclipse.DebouncedRunnable;
import com.tabbyml.tabby4eclipse.Logger;
import com.tabbyml.tabby4eclipse.StringUtils;
import com.tabbyml.tabby4eclipse.Utils;
import com.tabbyml.tabby4eclipse.chat.ChatViewUtils.RangeStrategy;
import com.tabbyml.tabby4eclipse.editor.EditorUtils;
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
	private Config.ServerConfig currentConfig;

	private List<String> pendingScripts = new ArrayList<>();
	private Map<String, CompletableFuture<Object>> pendingChatPanelRequest = new HashMap<>();

	private boolean isDark;
	private RGB bgColor;
	private RGB bgActiveColor;
	private RGB fgColor;
	private RGB borderColor;
	private RGB inputBorderColor;
	private RGB primaryColor;
	private RGB primaryFgColor;
	private RGB popoverColor;
	private RGB popoverFgColor;
	private RGB accentColor;
	private RGB accentFgColor;
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

		injectFunctions();
		load();
		serverConfigHolder.addConfigDidChangeListener(() -> {
			reloadContent(false);
		});
		statusInfoHolder.addStatusDidChangeListener(() -> {
			reloadContent(false);
		});
		
		PlatformUI.getWorkbench().getActiveWorkbenchWindow().getSelectionService().addSelectionListener(new ISelectionListener() {
			@Override
            public void selectionChanged(IWorkbenchPart part, ISelection selection) {
                if (selection instanceof ITextSelection) {
                	syncActiveSelectionRunnable.call();
                }
            }
		});
	}

	private DebouncedRunnable syncActiveSelectionRunnable = new DebouncedRunnable(() -> {
		if (!isChatPanelLoaded) {
			return;
		}
		EditorUtils.asyncExec(() -> {
			try {
				chatPanelClientInvoke("updateActiveSelection", new ArrayList<>() {
					{
						add(ChatViewUtils.getActiveEditorFileContext());
					}
				});
			} catch (Exception e) {
				// ignore
			}
		});
	}, 100);

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
		chatPanelClientInvoke("executeCommand", new ArrayList<>() {
			{
				add(ChatCommand.EXPLAIN);
			}
		});
	}

	public void fixSelectedText() {
		// FIXME(@icycodes): collect the diagnostic message provided by IDE or LSP
		chatPanelClientInvoke("executeCommand", new ArrayList<>() {
			{
				add(ChatCommand.FIX);
			}
		});
	}

	public void generateDocsForSelectedText() {
		chatPanelClientInvoke("executeCommand", new ArrayList<>() {
			{
				add(ChatCommand.GENERATE_DOCS);
			}
		});
	}

	public void generateTestsForSelectedText() {
		chatPanelClientInvoke("executeCommand", new ArrayList<>() {
			{
				add(ChatCommand.GENERATE_TESTS);
			}
		});
	}

	public void addSelectedTextAsContext() {
		chatPanelClientInvoke("addRelevantContext", new ArrayList<>() {
			{
				add(ChatViewUtils.getActiveEditorFileContext(RangeStrategy.SELECTION));
			}
		});
	}

	public void addActiveEditorAsContext() {
		chatPanelClientInvoke("addRelevantContext", new ArrayList<>() {
			{
				add(ChatViewUtils.getActiveEditorFileContext(RangeStrategy.FILE));
			}
		});
	}

	private void setupThemeStyle() {
		ITheme currentTheme = PlatformUI.getWorkbench().getThemeManager().getCurrentTheme();
		ColorRegistry colorRegistry = currentTheme.getColorRegistry();
		bgColor = colorRegistry.getRGB("org.eclipse.ui.workbench.ACTIVE_TAB_BG_START");
		isDark = (bgColor.red + bgColor.green + bgColor.blue) / 3 < 128;

		bgActiveColor = colorRegistry.getRGB("org.eclipse.ui.workbench.ACTIVE_TAB_BG_END");
		fgColor = colorRegistry.getRGB("org.eclipse.ui.workbench.ACTIVE_TAB_TEXT_COLOR");
		borderColor = isDark ? new RGB(64, 64, 64) : new RGB(192, 192, 192);
		inputBorderColor = borderColor;

		primaryColor = colorRegistry.getRGB("org.eclipse.ui.workbench.LINK_COLOR");
		if (primaryColor == null) {
			primaryColor = isDark ? new RGB(55, 148, 255) : new RGB(26, 133, 255);
		}
		primaryFgColor = new RGB(255, 255, 255);
		popoverColor = bgActiveColor;
		popoverFgColor = fgColor;
		accentColor = isDark ? new RGB(4, 57, 94)
				: new RGB((int) (bgActiveColor.red * 0.8), (int) (bgActiveColor.green * 0.8),
						(int) (bgActiveColor.blue * 0.8));
		accentFgColor = fgColor;

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
		if (inputBorderColor != null) {
			css += String.format("--input: %s;", StringUtils.toHsl(inputBorderColor));
		}
		if (primaryColor != null) {
			css += String.format("--primary: %s;", StringUtils.toHsl(primaryColor));
		}
		if (primaryFgColor != null) {
			css += String.format("--primary-foreground: %s;", StringUtils.toHsl(primaryFgColor));
		}
		if (popoverColor != null) {
			css += String.format("--popover: %s;", StringUtils.toHsl(popoverColor));
		}
		if (popoverFgColor != null) {
			css += String.format("--popover-foreground: %s;", StringUtils.toHsl(popoverFgColor));
		}
		if (accentColor != null) {
			css += String.format("--accent: %s;", StringUtils.toHsl(accentColor));
		}
		if (accentFgColor != null) {
			css += String.format("--accent-foreground: %s;", StringUtils.toHsl(accentFgColor));
		}
		if (font != null) {
			css += String.format("font: %s;", font);
		}
		css += String.format("font-size: %spt;", fontSize);
		return css;
	}

	private List<Object> parseArguments(final Object[] arguments) {
		if (arguments.length < 1) {
			return List.of();
		}
		return gson.fromJson(arguments[0].toString(), new TypeToken<List<Object>>() {
		});
	}

	private Object serializeResult(final Object result) {
		return gson.toJson(result);
	}

	private void injectFunctions() {
		browserFunctions.add(new BrowserFunction(browser, "handleTabbyChatPanelResponse") {
			@Override
			public Object function(Object[] arguments) {
				List<Object> params = parseArguments(arguments);
				logger.debug("Response from chat panel: " + params);
				if (params.size() < 3) {
					return null;
				}
				String uuid = (String) params.get(0);
				String errorMessage = (String) params.get(1);
				Object result = params.get(2);

				CompletableFuture<Object> future = pendingChatPanelRequest.remove(uuid);
				if (future == null) {
					return null;
				}

				if (errorMessage != null && !errorMessage.isEmpty()) {
					future.completeExceptionally(new Exception(errorMessage));
				} else {
					future.complete(result);
				}
				return null;
			}
		});

		browserFunctions.add(new BrowserFunction(browser, "handleReload") {
			@Override
			public Object function(Object[] arguments) {
				logger.debug("handleReload");
				reloadContent(true);
				return null;
			}
		});

		browserFunctions.add(new BrowserFunction(browser, "tabbyChatPanelRefresh") {
			@Override
			public Object function(Object[] arguments) {
				logger.debug("tabbyChatPanelRefresh");
				reloadContent(true);
				return null;
			}
		});

		browserFunctions.add(new BrowserFunction(browser, "tabbyChatPanelOnApplyInEditor") {
			@Override
			public Object function(Object[] arguments) {
				List<Object> params = parseArguments(arguments);
				logger.debug("tabbyChatPanelOnApplyInEditor: " + params);
				if (params.size() < 1) {
					return null;
				}
				String content = (String) params.get(0);
				ChatViewUtils.applyContentInEditor(content);
				return null;
			}
		});

		browserFunctions.add(new BrowserFunction(browser, "tabbyChatPanelOnLoaded") {
			@Override
			public Object function(Object[] arguments) {
				List<Object> params = parseArguments(arguments);
				logger.debug("tabbyChatPanelOnLoaded: " + params);
				if (params.size() < 1) {
					return null;
				}
				Map<String, Object> onLoadedParams = (Map<String, Object>) params.get(0);
				String apiVersion = (String) onLoadedParams.getOrDefault("apiVersion", "");
				if (!apiVersion.isBlank()) {
					String error = ChatViewUtils.checkChatPanelApiVersion(apiVersion);
					if (error != null) {
						updateContentToMessage(error);
						return null;
					}
				}
				initChatPanel();
				return null;
			}
		});

		browserFunctions.add(new BrowserFunction(browser, "tabbyChatPanelOnCopy") {
			@Override
			public Object function(Object[] arguments) {
				List<Object> params = parseArguments(arguments);
				logger.debug("tabbyChatPanelOnCopy: " + params);
				if (params.size() < 1) {
					return null;
				}
				String content = (String) params.get(0);
				ChatViewUtils.setClipboardContent(content);
				return null;
			}
		});

		browserFunctions.add(new BrowserFunction(browser, "tabbyChatPanelOnKeyboardEvent") {
			@Override
			public Object function(Object[] arguments) {
				// FIXME: For macOS and windows, the eclipse keyboard shortcuts are not
				// available when browser is focused,
				// we should handle keyboard events here.
				return null;
			}
		});

		browserFunctions.add(new BrowserFunction(browser, "tabbyChatPanelOpenInEditor") {
			@Override
			public Object function(Object[] arguments) {
				List<Object> params = parseArguments(arguments);
				logger.debug("tabbyChatPanelOpenInEditor: " + params);
				if (params.size() < 1) {
					return null;
				}
				FileLocation fileLocation = ChatViewUtils.asFileLocation(params.get(0));
				boolean success = ChatViewUtils.openInEditor(fileLocation);
				Object result = serializeResult(success);
				logger.debug("tabbyChatPanelOpenInEditor result: " + result);
				return result;
			}
		});

		browserFunctions.add(new BrowserFunction(browser, "tabbyChatPanelOpenExternal") {
			@Override
			public Object function(Object[] arguments) {
				List<Object> params = parseArguments(arguments);
				logger.debug("tabbyChatPanelOpenExternal: " + params);
				if (params.size() < 1) {
					return null;
				}
				String url = (String) params.get(0);
				ChatViewUtils.openExternal(url);
				return null;
			}
		});
		
		browserFunctions.add(new BrowserFunction(browser, "tabbyChatPanelReadWorkspaceGitRepositories") {
			@Override
			public Object function(Object[] arguments) {
				logger.debug("tabbyChatPanelReadWorkspaceGitRepositories");
				List<GitRepository> repositories = ChatViewUtils.readGitRepositoriesInWorkspace();
				Object result = serializeResult(repositories);
				logger.debug("tabbyChatPanelReadWorkspaceGitRepositories result: " + result);
				return result;
			}
		});

		browserFunctions.add(new BrowserFunction(browser, "tabbyChatPanelGetActiveEditorSelection") {
			@Override
			public Object function(Object[] arguments) {
				logger.debug("tabbyChatPanelGetActiveEditorSelection");
				EditorFileContext context = ChatViewUtils.getActiveEditorFileContext();
				Object result = serializeResult(context);
				logger.debug("tabbyChatPanelGetActiveEditorSelection result: " + result);
				return result;
			}
		});
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
		createChatPanelClient();
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
			currentConfig = null;
			updateContentToMessage("Cannot connect to Tabby server, please check your settings.");
		} else if (status.equals(StatusInfo.Status.CONNECTING)) {
			currentConfig = null;
			updateContentToMessage("Connecting to Tabby server...");
		} else if (status.equals(StatusInfo.Status.UNAUTHORIZED)) {
			currentConfig = null;
			updateContentToMessage("Authorization required, please set your token in settings.");
		} else {
			Map<String, Object> serverHealth = statusInfoHolder.getStatusInfo().getServerHealth();
			String error = ChatViewUtils.checkServerHealth(serverHealth);
			if (error != null) {
				currentConfig = null;
				updateContentToMessage(error);
			} else {
				// Load main
				Config.ServerConfig config = serverConfigHolder.getConfig().getServer();
				if (config == null) {
					currentConfig = null;
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

	// execute js functions

	private void executeScript(String script) {
		browser.getDisplay().asyncExec(() -> {
			browser.execute(script);
		});
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

	private void initChatPanel() {
		isChatPanelLoaded = true;
		browser.getDisplay().timerExec(100, () -> {
			updateContentToChatPanel();
			pendingScripts.forEach((script) -> {
				executeScript(script);
			});
			pendingScripts.clear();
		});
		chatPanelClientInvoke("init", new ArrayList<>() {
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
		});
		chatPanelClientInvoke("updateTheme", new ArrayList<>() {
			{
				add(buildCss());
				add(isDark ? "dark" : "light");
			}
		});
	}

	private String wrapJsFunction(String name) {
		return String.format(
			String.join("\n",
				"function(...args) {",
				"  return new Promise((resolve, reject) => {",
				"    const paramsJson = JSON.stringify(args)",
				"    const result = %s(paramsJson)",
				"    resolve(JSON.parse(result))",
				"  });",
				"}"
			),
			name
		);
	}

	private void createChatPanelClient() {
		String script = String.format(
			String.join("\n",
				"if (!window.tabbyChatPanelClient) {",
				"  window.tabbyChatPanelClient = TabbyThreads.createThreadFromIframe(getChatPanel(), {",
				"    expose: {",
				"      refresh: %s,",
				"      onApplyInEditor: %s,",
				"      onLoaded: %s,",
				"      onCopy: %s,",
				"      onKeyboardEvent: %s,",
				"      openInEditor: %s,",
				"      openExternal: %s,",
				"      readWorkspaceGitRepositories: %s,",
				"      getActiveEditorSelection: %s,",
				"    }",
				"  })",
				"}"
			),
			wrapJsFunction("tabbyChatPanelRefresh"),
			wrapJsFunction("tabbyChatPanelOnApplyInEditor"),
			wrapJsFunction("tabbyChatPanelOnLoaded"),
			wrapJsFunction("tabbyChatPanelOnCopy"),
			wrapJsFunction("tabbyChatPanelOnKeyboardEvent"),
			wrapJsFunction("tabbyChatPanelOpenInEditor"),
			wrapJsFunction("tabbyChatPanelOpenExternal"),
			wrapJsFunction("tabbyChatPanelReadWorkspaceGitRepositories"),
			wrapJsFunction("tabbyChatPanelGetActiveEditorSelection")
		);
		executeScript(script);
	}

	private CompletableFuture<Object> chatPanelClientInvoke(String method, List<Object> params) {
		CompletableFuture<Object> future = new CompletableFuture<>();
		String uuid = UUID.randomUUID().toString();
		pendingChatPanelRequest.put(uuid, future);
		String paramsJson = StringUtils.escapeCharacters(gson.toJson(params));
		String responseCallbackFunction = "handleTabbyChatPanelResponse(results)";
		String script = String.format(
			String.join("\n",
				"(function() {",
				"  const func = window.tabbyChatPanelClient['%s']",
				"  if (func && typeof func === 'function') {",
				"    const params = JSON.parse('%s')",
				"    const resultPromise = func(...params)",
				"    if (resultPromise && typeof resultPromise.then === 'function') {",
				"      resultPromise.then(result => {",
				"        const results = JSON.stringify(['%s', null, result])",
				"        %s",
				"      }).catch(error => {",
				"        const results = JSON.stringify(['%s', error.message, null])",
				"        %s",
				"      })",
				"    } else {",
				"      const results = JSON.stringify(['%s', null, resultPromise])",
				"      %s",
				"    }",
				"  } else {",
				"    const results = JSON.stringify(['%s', 'Method not found: %s', null])",
				"    %s",
				"  }",
				"})()"
			),
			method,
			paramsJson,
			uuid,
			responseCallbackFunction,
			uuid,
			responseCallbackFunction,
			uuid,
			responseCallbackFunction,
			uuid,
			method,
			responseCallbackFunction
		);
		logger.debug("Request to chat panel: " + uuid + ", " + method + ", " + paramsJson);
		if (isChatPanelLoaded) {
			executeScript(script);
		} else {
			pendingScripts.add(script);
		}
		return future;
	}
}
