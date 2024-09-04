package com.tabbyml.tabby4eclipse.chat;

import java.net.URI;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.eclipse.core.resources.IFile;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.ResourcesPlugin;
import org.eclipse.core.runtime.FileLocator;
import org.eclipse.core.runtime.Path;
import org.eclipse.core.runtime.Platform;
import org.eclipse.jface.resource.ColorRegistry;
import org.eclipse.jface.resource.FontRegistry;
import org.eclipse.jface.text.IDocument;
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
import org.eclipse.ui.IEditorPart;
import org.eclipse.ui.IWorkbenchPage;
import org.eclipse.ui.IWorkbenchWindow;
import org.eclipse.ui.PartInitException;
import org.eclipse.ui.PlatformUI;
import org.eclipse.ui.ide.IDE;
import org.eclipse.ui.ide.ResourceUtil;
import org.eclipse.ui.part.ViewPart;
import org.eclipse.ui.texteditor.ITextEditor;
import org.eclipse.ui.themes.ITheme;
import org.osgi.framework.Bundle;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.tabbyml.tabby4eclipse.Activator;
import com.tabbyml.tabby4eclipse.Logger;
import com.tabbyml.tabby4eclipse.Utils;
import com.tabbyml.tabby4eclipse.chat.ChatMessage.FileContext;
import com.tabbyml.tabby4eclipse.editor.EditorUtils;
import com.tabbyml.tabby4eclipse.git.GitProvider;
import com.tabbyml.tabby4eclipse.lsp.LanguageServerService;
import com.tabbyml.tabby4eclipse.lsp.ServerConfigHolder;
import com.tabbyml.tabby4eclipse.lsp.StatusInfoHolder;
import com.tabbyml.tabby4eclipse.lsp.protocol.Config;
import com.tabbyml.tabby4eclipse.lsp.protocol.GitRepository;
import com.tabbyml.tabby4eclipse.lsp.protocol.GitRepositoryParams;
import com.tabbyml.tabby4eclipse.lsp.protocol.ILanguageServer;
import com.tabbyml.tabby4eclipse.lsp.protocol.IStatusService;
import com.tabbyml.tabby4eclipse.lsp.protocol.StatusInfo;
import com.tabbyml.tabby4eclipse.lsp.protocol.StatusRequestParams;

public class ChatView extends ViewPart {
	private static final String MIN_SERVER_VERSION = "0.16.0";
	private static final String ID = "com.tabbyml.tabby4eclipse.views.chat";

	public static void openChatView() {
		IWorkbenchWindow workbenchWindow = PlatformUI.getWorkbench().getActiveWorkbenchWindow();
		if (workbenchWindow != null) {
			IWorkbenchPage page = workbenchWindow.getActivePage();
			if (page != null) {
				try {
					page.showView(ID);
				} catch (PartInitException e) {
					e.printStackTrace();
				}
			}
		}
	}

	private Logger logger = new Logger("ChatView");
	private Gson gson = new Gson();

	private StatusInfoHolder statusInfoHolder = StatusInfoHolder.getInstance();
	private ServerConfigHolder serverConfigHolder = ServerConfigHolder.getInstance();

	private Browser browser;
	private List<BrowserFunction> browserFunctions = new ArrayList<>();

	private boolean isHtmlLoaded = false;
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

		browser = new Browser(parent, Utils.isWindows() ? SWT.EDGE : SWT.DEFAULT);
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

		browserFunctions.add(new BrowserFunction(browser, "handleChatPanelLoaded") {
			@Override
			public Object function(Object[] arguments) {
				handleChatPanelLoaded();
				return null;
			}
		});

		browserFunctions.add(new BrowserFunction(browser, "handleChatPanelStyleApplied") {
			@Override
			public Object function(Object[] arguments) {
				handleChatPanelStyleApplied();
				return null;
			}
		});

		browserFunctions.add(new BrowserFunction(browser, "handleChatPanelRequest") {
			@Override
			public Object function(Object[] arguments) {
				if (arguments.length > 0) {
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
		browser.setFocus();
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
			showMessage("Cannot connect to Tabby server, please check your settings.");
			showChatPanel(false);
		} else if (status.equals(StatusInfo.Status.UNAUTHORIZED)) {
			showMessage("Authorization required, please set your token in settings.");
			showChatPanel(false);
		} else {
			Map<String, Object> serverHealth = statusInfoHolder.getStatusInfo().getServerHealth();
			String error = checkServerHealth(serverHealth);
			if (error != null) {
				showMessage(error);
				showChatPanel(false);
			} else {
				// Load main
				Config.ServerConfig config = serverConfigHolder.getConfig().getServer();
				if (config != null
						&& (force || currentConfig == null || currentConfig.getEndpoint() != config.getEndpoint()
								|| currentConfig.getToken() != config.getToken())) {
					showMessage("Connecting to Tabby server...");
					showChatPanel(false);
					currentConfig = config;
					loadChatPanel();
				}
			}
		}
	}

	private void showMessage(String message) {
		browser.getDisplay().asyncExec(() -> {
			if (message != null) {
				browser.execute(String.format("showMessage('%s')", message));
			} else {
				browser.execute("showMessage(undefined)");
			}
		});
	}

	private void showChatPanel(boolean visiable) {
		browser.getDisplay().asyncExec(() -> {
			browser.execute(String.format("showChatPanel(%s)", visiable ? "true" : "false"));
		});
	}

	private void loadChatPanel() {
		// FIXME(@icycodes): set query string to vscode for now to turn on callbacks
		String chatUrl = String.format("%s/chat?client=vscode", currentConfig.getEndpoint());
		browser.getDisplay().asyncExec(() -> {
			browser.execute(String.format("loadChatPanel('%s')", chatUrl));
		});
	}

	private String checkServerHealth(Map<String, Object> serverHealth) {
		if (serverHealth == null) {
			return "Connecting to Tabby server...";
		}

		if (serverHealth.get("webserver") == null || serverHealth.get("chat_model") == null) {
			return "You need to launch the server with the chat model enabled; for example, use `--chat-model Qwen2-1.5B-Instruct`.";
		}

		if (serverHealth.containsKey("version")) {
			String version = null;
			Object versionObj = serverHealth.get("version");
			if (versionObj instanceof String versionStr) {
				version = versionStr;
			} else if (versionObj instanceof Map versionMap) {
				if (versionMap.containsKey("git_describe")
						&& versionMap.get("git_describe") instanceof String versionStr) {
					version = versionStr;
				}
			}
			if (version != null && !isVersionCompatible(version)) {
				return String.format(
						"Tabby Chat requires Tabby server version %s or later. Your server is running version %s.",
						MIN_SERVER_VERSION, version);
			}
		}
		return null;
	}

	private boolean isVersionCompatible(String version) {
		String versionStr = version;
		if (versionStr != null && versionStr.length() > 0 && versionStr.charAt(0) == 'v') {
			versionStr = versionStr.substring(1);
		}
		String[] versionParts = versionStr.trim().split("\\.");
		String[] minVersionParts = MIN_SERVER_VERSION.split("\\.");

		for (int i = 0; i < Math.max(versionParts.length, minVersionParts.length); i++) {
			int versionPart = i < versionParts.length ? parseInt(versionParts[i]) : 0;
			int minVersionPart = i < minVersionParts.length ? parseInt(minVersionParts[i]) : 0;

			if (versionPart < minVersionPart) {
				return false;
			} else if (versionPart > minVersionPart) {
				return true;
			}
		}

		return true;
	}

	private int parseInt(String str) {
		try {
			return Integer.parseInt(str);
		} catch (NumberFormatException e) {
			return 0;
		}
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

	private void applyStyle() {
		String theme = isDark ? "dark" : "light";
		String css = buildCss();
		String json = gson.toJson(new HashMap<>() {
			{
				put("theme", theme);
				put("css", css);
			}
		});
		browser.getDisplay().asyncExec(() -> {
			browser.execute(String.format("applyStyle('%s')", json));
			browser.setVisible(true);
		});
	}

	private String buildCss() {
		String css = "";
		if (bgActiveColor != null) {
			css += String.format("background-color: hsl(%s);", toHsl(bgActiveColor));
		}
		if (bgColor != null) {
			css += String.format("--background: %s;", toHsl(bgColor));
		}
		if (fgColor != null) {
			css += String.format("--foreground: %s;", toHsl(fgColor));
		}
		if (borderColor != null) {
			css += String.format("--border: %s;", toHsl(borderColor));
		}
		if (primaryColor != null) {
			css += String.format("--primary: %s;", toHsl(primaryColor));
		}
		if (font != null) {
			css += String.format("font: %s;", font);
		}
		css += String.format("font-size: %spt;", fontSize);
		return css;
	}

	private static String toHsl(RGB rgb) {
		double r = rgb.red / 255.0;
		double g = rgb.green / 255.0;
		double b = rgb.blue / 255.0;
		double max = Math.max(r, Math.max(g, b));
		double min = Math.min(r, Math.min(g, b));
		double l = (max + min) / 2.0;
		double h, s;
		if (max == min) {
			h = 0;
			s = 0;
		} else {
			double delta = max - min;
			s = l > 0.5 ? delta / (2.0 - max - min) : delta / (max + min);
			if (max == r) {
				h = (g - b) / delta + (g < b ? 6 : 0);
			} else if (max == g) {
				h = (b - r) / delta + 2;
			} else {
				h = (r - g) / delta + 4;
			}
			h /= 6;
		}
		h *= 360;
		s *= 100;
		l *= 100;
		return String.format("%.0f, %.0f%%, %.0f%%", h, s, l);
	}

	private void handleChatPanelLoaded() {
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
	}

	private void handleChatPanelStyleApplied() {
		showMessage(null);
		showChatPanel(true);
	}

	private void sendRequestToChatPanel(Request request) {
		String json = gson.toJson(request);
		browser.getDisplay().asyncExec(() -> {
			browser.execute(String.format("sendRequestToChatPanel('%s')", escapeCharacters(json)));
		});
	}

	public static String escapeCharacters(String jsonString) {
		return jsonString.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r")
				.replace("\t", "\\t");
	}

	private void handleChatPanelRequest(Request request) {
		switch (request.getMethod()) {
		case "navigate": {
			List<Object> params = request.getParams();
			if (params.size() < 1) {
				return;
			}
			FileContext context = gson.fromJson(gson.toJson(params.get(0)), FileContext.class);
			navigateToFileContext(context);
			break;
		}
		case "onSubmitMessage": {
			List<Object> params = request.getParams();
			if (params.size() < 1) {
				return;
			}
			String message = (String) params.get(0);
			List<FileContext> releventContexts = params.size() > 1
					? releventContexts = gson.fromJson(gson.toJson(params.get(1)), new TypeToken<List<FileContext>>() {
					}.getType())
					: null;
			sendRequestToChatPanel(new Request("sendMessage", new ArrayList<>() {
				{
					ChatMessage chatMessage = new ChatMessage();
					chatMessage.setMessage(message);
					if (releventContexts != null && !releventContexts.isEmpty()) {
						chatMessage.setRelevantContext(releventContexts);
					} else {
						chatMessage.setActiveContext(getActiveContext());
					}
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
			applyContentInEditor(content);
			break;
		}
		}
	}

	private FileContext getActiveContext() {
		ITextEditor activeTextEditor = EditorUtils.getActiveTextEditor();
		if (activeTextEditor == null) {
			return null;
		}
		FileContext context = new FileContext();

		IFile file = ResourceUtil.getFile(activeTextEditor.getEditorInput());
		URI fileUri = file.getLocationURI();
		if (file != null) {
			GitRepository gitInfo = GitProvider.getInstance()
					.getRepository(new GitRepositoryParams(fileUri.toString()));
			IProject project = file.getProject();
			if (gitInfo != null) {
				try {
					context.setGitUrl(gitInfo.getRemoteUrl());
					String relativePath = new URI(gitInfo.getRoot()).relativize(fileUri).getPath();
					context.setFilePath(relativePath);
				} catch (Exception e) {
					logger.error("Failed to get git info.", e);
				}
			} else if (project != null) {
				URI projectRoot = project.getLocationURI();
				String relativePath = projectRoot.relativize(fileUri).getPath();
				context.setFilePath(relativePath);
			} else {
				context.setFilePath(fileUri.toString());
			}
		}

		ISelection selection = activeTextEditor.getSelectionProvider().getSelection();
		if (selection instanceof ITextSelection textSelection) {
			if (textSelection.isEmpty() || textSelection.getText().isBlank()) {
				IDocument document = activeTextEditor.getDocumentProvider()
						.getDocument(activeTextEditor.getEditorInput());
				context.setRange(new FileContext.LineRange(1, document.getNumberOfLines()));
				context.setContent(document.get());
			} else {
				context.setRange(
						new FileContext.LineRange(textSelection.getStartLine() + 1, textSelection.getEndLine() + 1));
				context.setContent(textSelection.getText());
			}
		}
		return context;
	}

	private void navigateToFileContext(FileContext context) {
		logger.info("Navigate to file: " + context.getFilePath() + ", line: " + context.getRange().getStart());
		// FIXME(@icycode): the base path could be a git repository root, but it cannot
		// be determined here
		IFile file = null;
		ITextEditor activeTextEditor = EditorUtils.getActiveTextEditor();
		if (activeTextEditor != null) {
			// try find file in the project of the active editor
			IFile activeFile = ResourceUtil.getFile(activeTextEditor.getEditorInput());
			if (activeFile != null) {
				file = activeFile.getProject().getFile(new Path(context.getFilePath()));
			}
		} else {
			// try find file in the workspace
			file = ResourcesPlugin.getWorkspace().getRoot().getFileForLocation(new Path(context.getFilePath()));
		}
		try {
			if (file != null && file.exists()) {
				IEditorPart editorPart = IDE.openEditor(EditorUtils.getActiveWorkbenchPage(), file);
				if (editorPart instanceof ITextEditor textEditor) {
					IDocument document = textEditor.getDocumentProvider().getDocument(textEditor.getEditorInput());
					int offset = document.getLineOffset(context.getRange().getStart() - 1);
					textEditor.selectAndReveal(offset, 0);
				}
			}
		} catch (Exception e) {
			logger.error("Failed to navigate to file: " + context.getFilePath(), e);
		}
	}

	private void applyContentInEditor(String content) {
		logger.info("Apply content to the active text editor.");
		ITextEditor activeTextEditor = EditorUtils.getActiveTextEditor();
		if (activeTextEditor != null) {
			try {
				IDocument document = activeTextEditor.getDocumentProvider()
						.getDocument(activeTextEditor.getEditorInput());
				ITextSelection selection = (ITextSelection) activeTextEditor.getSelectionProvider().getSelection();
				document.replace(selection.getOffset(), selection.getLength(), content);
			} catch (Exception e) {
				logger.error("Failed to apply content to the active text editor.", e);
			}
		}
	}
}
