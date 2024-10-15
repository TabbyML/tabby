package com.tabbyml.tabby4eclipse.chat;

import java.net.URI;
import java.util.Map;

import org.eclipse.core.resources.IFile;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.ResourcesPlugin;
import org.eclipse.core.runtime.Path;
import org.eclipse.jface.text.IDocument;
import org.eclipse.jface.text.ITextSelection;
import org.eclipse.jface.viewers.ISelection;
import org.eclipse.swt.dnd.Clipboard;
import org.eclipse.swt.dnd.TextTransfer;
import org.eclipse.swt.dnd.Transfer;
import org.eclipse.swt.widgets.Display;
import org.eclipse.ui.IEditorPart;
import org.eclipse.ui.IWorkbenchPage;
import org.eclipse.ui.PartInitException;
import org.eclipse.ui.ide.IDE;
import org.eclipse.ui.ide.ResourceUtil;
import org.eclipse.ui.texteditor.ITextEditor;

import com.tabbyml.tabby4eclipse.Logger;
import com.tabbyml.tabby4eclipse.Version;
import com.tabbyml.tabby4eclipse.chat.ChatMessage.FileContext;
import com.tabbyml.tabby4eclipse.editor.EditorUtils;
import com.tabbyml.tabby4eclipse.git.GitProvider;
import com.tabbyml.tabby4eclipse.lsp.protocol.GitRepository;
import com.tabbyml.tabby4eclipse.lsp.protocol.GitRepositoryParams;

public class ChatViewUtils {
	private static final String ID = "com.tabbyml.tabby4eclipse.views.chat";

	private static final String MIN_SERVER_VERSION = "0.18.0";
	private static final String CHAT_PANEL_API_VERSION = "0.2.0";
	private static Logger logger = new Logger("ChatView");

	public static final String PROMPT_EXPLAIN = "Explain the selected code:";
	public static final String PROMPT_FIX = "Identify and fix potential bugs in the selected code:";
	public static final String PROMPT_GENERATE_DOCS = "Generate documentation for the selected code:";
	public static final String PROMPT_GENERATE_TESTS = "Generate a unit test for the selected code:";

	public static ChatView openChatView() {
		IWorkbenchPage page = EditorUtils.getActiveWorkbenchPage();
		if (page != null) {
			try {
				page.showView(ID);
				return (ChatView) page.findView(ID);
			} catch (PartInitException e) {
				logger.error("Failed to open chat view.", e);
			}
		}
		return null;
	}

	public static ChatView findOpenedView() {
		IWorkbenchPage page = EditorUtils.getActiveWorkbenchPage();
		if (page != null && page.findView(ID) instanceof ChatView chatView) {
			return chatView;
		}
		return null;
	}

	public static String checkServerHealth(Map<String, Object> serverHealth) {
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
			if (version != null) {
				Version parsedVersion = new Version(version);
				Version requiredVersion = new Version(MIN_SERVER_VERSION);
				if (!parsedVersion.isGreaterOrEqualThan(requiredVersion)) {
					return String.format(
							"Tabby Chat requires Tabby server version %s or later. Your server is running version %s.",
							MIN_SERVER_VERSION, version);
				}
			}
		}
		return null;
	}

	public static String checkChatPanelApiVersion(String version) {
		Version parsedVersion = new Version(version);
		Version requiredVersion = new Version(CHAT_PANEL_API_VERSION);
		if (!parsedVersion.isEqual(requiredVersion, true)) {
			return "Please update your Tabby server and Tabby plugin for Eclipse to the latest version to use chat panel.";
		}
		return null;
	}

	public static FileContext getSelectedTextAsFileContext() {
		ITextEditor activeTextEditor = EditorUtils.getActiveTextEditor();
		if (activeTextEditor == null) {
			return null;
		}
		FileContext context = new FileContext();
		ISelection selection = activeTextEditor.getSelectionProvider().getSelection();
		if (selection instanceof ITextSelection textSelection) {
			if (!textSelection.isEmpty()) {
				String content = textSelection.getText();
				if (!content.isBlank()) {
					context.setContent(content);
					context.setRange(new FileContext.LineRange(textSelection.getStartLine() + 1,
							textSelection.getEndLine() + 1));
				}
			}
		}
		if (context.getContent() == null) {
			return null;
		}

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
		return context;
	}

	public static FileContext getActiveEditorAsFileContext() {
		ITextEditor activeTextEditor = EditorUtils.getActiveTextEditor();
		if (activeTextEditor == null) {
			return null;
		}
		FileContext context = new FileContext();

		IDocument document = EditorUtils.getDocument(activeTextEditor);
		context.setRange(new FileContext.LineRange(1, document.getNumberOfLines()));
		context.setContent(document.get());

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

		return context;
	}

	public static void navigateToFileContext(FileContext context) {
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

	public static void setClipboardContent(String content) {
		Display display = Display.getCurrent();
		if (display == null) {
			display = Display.getDefault();
		}

		Clipboard clipboard = new Clipboard(display);
		TextTransfer textTransfer = TextTransfer.getInstance();
		clipboard.setContents(new Object[] { content }, new Transfer[] { textTransfer });
		clipboard.dispose();
	}

	public static void applyContentInEditor(String content) {
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
