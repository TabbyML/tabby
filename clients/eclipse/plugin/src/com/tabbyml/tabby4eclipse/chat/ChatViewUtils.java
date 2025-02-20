package com.tabbyml.tabby4eclipse.chat;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.eclipse.core.resources.IFile;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.IWorkspaceRoot;
import org.eclipse.core.resources.ResourcesPlugin;
import org.eclipse.core.runtime.Path;
import org.eclipse.jface.text.IDocument;
import org.eclipse.jface.text.ITextSelection;
import org.eclipse.jface.viewers.ISelection;
import org.eclipse.swt.dnd.Clipboard;
import org.eclipse.swt.dnd.TextTransfer;
import org.eclipse.swt.dnd.Transfer;
import org.eclipse.swt.program.Program;
import org.eclipse.swt.widgets.Display;
import org.eclipse.ui.IEditorPart;
import org.eclipse.ui.IWorkbenchPage;
import org.eclipse.ui.PartInitException;
import org.eclipse.ui.ide.IDE;
import org.eclipse.ui.ide.ResourceUtil;
import org.eclipse.ui.texteditor.ITextEditor;

import com.google.gson.Gson;
import com.tabbyml.tabby4eclipse.Logger;
import com.tabbyml.tabby4eclipse.Version;
import com.tabbyml.tabby4eclipse.editor.EditorUtils;
import com.tabbyml.tabby4eclipse.git.GitProvider;
import com.tabbyml.tabby4eclipse.lsp.protocol.GitRepositoryParams;

public class ChatViewUtils {
	private static final String ID = "com.tabbyml.tabby4eclipse.views.chat";

	private static final String MIN_SERVER_VERSION = "0.25.0";
	private static final String CHAT_PANEL_API_VERSION = "0.7.0";
	private static Logger logger = new Logger("ChatView");

	private static final Gson gson = new Gson();
	private static final Map<String, String> gitRemoteUrlToLocalRoot = new HashMap<>();

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
				if (!parsedVersion.isZero() && !parsedVersion.isGreaterOrEqualThan(requiredVersion)) {
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

	// default: use selection if available, otherwise use the whole file
	// selection: use selection if available, otherwise return null
	// file: use the whole file
	public static enum RangeStrategy {
		DEFAULT, SELECTION, FILE
	}

	public static EditorFileContext getActiveEditorFileContext() {
		return getActiveEditorFileContext(RangeStrategy.DEFAULT);
	}
	
	public static EditorFileContext getActiveEditorFileContext(RangeStrategy rangeStrategy) {
		ITextEditor activeTextEditor = EditorUtils.getActiveTextEditor();
		if (activeTextEditor == null) {
			return null;
		}
		IFile file = ResourceUtil.getFile(activeTextEditor.getEditorInput());
		ISelection selection = activeTextEditor.getSelectionProvider().getSelection();
		boolean hasSelection = false;
		if (selection instanceof ITextSelection textSelection) {
			if (!textSelection.isEmpty()) {
				String content = textSelection.getText();
				if (!content.isBlank()) {
					hasSelection = true;
				}
			}
		}

		if (rangeStrategy == RangeStrategy.SELECTION || (rangeStrategy == RangeStrategy.DEFAULT && hasSelection)) {
			if (selection instanceof ITextSelection textSelection) {
				if (!textSelection.isEmpty()) {
					String content = textSelection.getText();
					if (!content.isBlank()) {
						return new EditorFileContext(fileToChatPanelFilepath(file),
								new LineRange(textSelection.getStartLine() + 1, textSelection.getEndLine() + 1),
								content);
					}
				}
			}
		} else {
			IDocument document = EditorUtils.getDocument(activeTextEditor);
			String content = document.get();
			if (!content.isBlank()) {
				return new EditorFileContext(fileToChatPanelFilepath(file), null, content);
			}
		}
		return null;
	}

	public static boolean openInEditor(FileLocation fileLocation) {
		if (fileLocation == null) {
			return false;
		}
		Filepath filepath = fileLocation.getFilepath();
		try {
			IFile file = chatPanelFilepathToFile(filepath);
			if (file != null && file.exists()) {
				IEditorPart editorPart = IDE.openEditor(EditorUtils.getActiveWorkbenchPage(), file);

				if (editorPart instanceof ITextEditor textEditor) {
					IDocument document = textEditor.getDocumentProvider().getDocument(textEditor.getEditorInput());
					Object location = fileLocation.getLocation();
					Position position;

					if (location instanceof Number lineNumberValue) {
						position = new Position(lineNumberValue.intValue() - 1, 0);
					} else if (location instanceof Position positionValue) {
						position = new Position(positionValue.getLine() - 1, positionValue.getCharacter() - 1);
					} else if (location instanceof LineRange lineRangeValue) {
						position = new Position(lineRangeValue.getStart() - 1, 0);
					} else if (location instanceof PositionRange positionRangeValue) {
						position = new Position(positionRangeValue.getStart().getLine() - 1,
								positionRangeValue.getStart().getCharacter() - 1);
					} else {
						position = null;
					}

					if (position != null) {
						int offset = document.getLineOffset(position.getLine()) + position.getCharacter();
						textEditor.selectAndReveal(offset, 0);
					}
				}
				return true;
			} else {
				return false;
			}
		} catch (Exception e) {
			logger.error("Failed to open in editor.", e);
			return false;
		}
	}

	public static void openExternal(String url) {
		Program.launch(url);
	}

	public static List<GitRepository> readGitRepositoriesInWorkspace() {
		List<GitRepository> repositories = new ArrayList<>();
		IWorkspaceRoot workspaceRoot = ResourcesPlugin.getWorkspace().getRoot();
		IProject[] projects = workspaceRoot.getProjects();

		for (IProject project : projects) {
			try {
				URI projectRootUri = project.getLocation().toFile().toURI();
				com.tabbyml.tabby4eclipse.lsp.protocol.GitRepository repo = GitProvider.getInstance()
						.getRepository(new GitRepositoryParams(projectRootUri.toString()));
				if (repo != null) {
					repositories.add(new GitRepository(repo.getRemoteUrl()));
				}
			} catch (Exception e) {
				logger.warn("Error when read git repository.", e);
			}
		}
		return repositories;
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

	public static Filepath fileToChatPanelFilepath(IFile file) {
		if (file == null) {
			return null;
		}
		URI fileUri = file.getLocationURI();
		String fileUriString = fileUri.toString();

		com.tabbyml.tabby4eclipse.lsp.protocol.GitRepository gitRepo = GitProvider.getInstance()
				.getRepository(new GitRepositoryParams(fileUriString));
		String gitUrl = (gitRepo != null) ? gitRepo.getRemoteUrl() : null;
		if (gitUrl != null) {
			gitRemoteUrlToLocalRoot.put(gitUrl, gitRepo.getRoot());
		}
		if (gitUrl != null && fileUriString.startsWith(gitRepo.getRoot())) {
			try {
				String relativePath = new URI(gitRepo.getRoot()).relativize(fileUri).getPath();
				return new FilepathInGitRepository(relativePath, gitUrl);
			} catch (Exception e) {
				// nothing
			}
		}

		IWorkspaceRoot workspaceRoot = ResourcesPlugin.getWorkspace().getRoot();
		IProject[] projects = workspaceRoot.getProjects();
		for (IProject project : projects) {
			try {
				URI projectRootUri = project.getLocation().toFile().toURI();
				String projectRootUriString = projectRootUri.toString();
				if (fileUriString.startsWith(projectRootUriString)) {
					String relativePath = projectRootUri.relativize(fileUri).getPath();
					return new FilepathInWorkspace(relativePath, projectRootUriString);
				}
			} catch (Exception e) {
				// nothing
			}
		}

		return new FilepathUri(fileUriString);
	}

	public static IFile chatPanelFilepathToFile(Filepath filepath) {
		IWorkspaceRoot workspaceRoot = ResourcesPlugin.getWorkspace().getRoot();

		switch (filepath.getKind()) {
		case Filepath.Kind.URI: {
			FilepathUri filepathUri = (FilepathUri) filepath;
			try {
				URI fileUri = new URI(filepathUri.getUri());
				IFile file = workspaceRoot.getFileForLocation(new Path(fileUri.getPath()));
				if (file != null && file.exists()) {
					return file;
				}
			} catch (URISyntaxException e) {
				IProject[] projects = workspaceRoot.getProjects();
				for (IProject project : projects) {
					URI projectRootUri = project.getLocation().toFile().toURI();
					URI fileUri = projectRootUri.resolve(filepathUri.getUri());
					IFile file = workspaceRoot.getFileForLocation(new Path(fileUri.getPath()));
					if (file != null && file.exists()) {
						return file;
					}
				}
			}
			break;
		}

		case Filepath.Kind.WORKSPACE: {
			FilepathInWorkspace filepathInWorkspace = (FilepathInWorkspace) filepath;
			try {
				URI fileUri = new URI(filepathInWorkspace.getBaseDir()).resolve(filepathInWorkspace.getFilepath());
				IFile file = workspaceRoot.getFileForLocation(new Path(fileUri.getPath()));
				if (file != null && file.exists()) {
					return file;
				}
			} catch (Exception e) {
				// nothing
			}
			break;
		}

		case Filepath.Kind.GIT:
			FilepathInGitRepository filepathInGit = (FilepathInGitRepository) filepath;
			String gitLocalRoot = gitRemoteUrlToLocalRoot.get(filepathInGit.getGitUrl());
			if (gitLocalRoot != null) {
				try {
					URI fileUri = new URI(gitLocalRoot).resolve(filepathInGit.getFilepath());
					IFile file = workspaceRoot.getFileForLocation(new Path(fileUri.getPath()));
					if (file != null && file.exists()) {
						return file;
					}
				} catch (Exception e) {
					// nothing
				}
			}
			break;

		default:
			break;
		}

		logger.warn("Failed to parse filepath: " + gson.toJson(filepath));
		return null;
	}

	public static FileLocation asFileLocation(Object obj) {
		if (!(obj instanceof Map)) {
			return null;
		}

		Map<?, ?> map = (Map<?, ?>) obj;

		if (!map.containsKey("filepath")) {
			return null;
		}

		Object filepathValue = map.get("filepath");
		Filepath filepath = null;

		if (filepathValue instanceof Map) {
			Map<?, ?> filepathMap = (Map<?, ?>) filepathValue;
			if (filepathMap.containsKey("kind")) {
				String kind = (String) filepathMap.get("kind");
				if (Filepath.Kind.GIT.equals(kind)) {
					filepath = gson.fromJson(gson.toJson(filepathValue), FilepathInGitRepository.class);
				} else if (Filepath.Kind.WORKSPACE.equals(kind)) {
					filepath = gson.fromJson(gson.toJson(filepathValue), FilepathInWorkspace.class);
				} else if (Filepath.Kind.URI.equals(kind)) {
					filepath = gson.fromJson(gson.toJson(filepathValue), FilepathUri.class);
				}
			}
		}

		if (filepath == null) {
			return null;
		}

		Object locationValue = map.get("location");
		Object location = null;

		if (locationValue instanceof Number) {
			location = locationValue;
		} else if (locationValue instanceof Map) {
			Map<?, ?> locationMap = (Map<?, ?>) locationValue;
			if (locationMap.containsKey("line")) {
				location = gson.fromJson(gson.toJson(locationValue), Position.class);
			} else if (locationMap.containsKey("start")) {
				Object startValue = locationMap.get("start");
				if (startValue instanceof Number) {
					location = gson.fromJson(gson.toJson(locationValue), LineRange.class);
				} else if (startValue instanceof Map) {
					location = gson.fromJson(gson.toJson(locationValue), PositionRange.class);
				}
			}
		}

		return new FileLocation(filepath, location);
	}
}
