package com.tabbyml.tabby4eclipse.editor;

import java.net.URI;
import java.util.ArrayList;
import java.util.List;

import org.eclipse.jface.text.IDocument;
import org.eclipse.lsp4e.LanguageServerWrapper;
import org.eclipse.ui.IEditorPart;
import org.eclipse.ui.IEditorReference;
import org.eclipse.ui.IPartListener;
import org.eclipse.ui.IWorkbenchPage;
import org.eclipse.ui.IWorkbenchPart;
import org.eclipse.ui.IWorkbenchWindow;
import org.eclipse.ui.PlatformUI;
import org.eclipse.ui.texteditor.ITextEditor;

import com.tabbyml.tabby4eclipse.Logger;
import com.tabbyml.tabby4eclipse.inlineCompletion.InlineCompletionTrigger;
import com.tabbyml.tabby4eclipse.lsp.LanguageServerService;

public class WorkbenchPartListener implements IPartListener {
	public static WorkbenchPartListener getInstance() {
		return LazyHolder.INSTANCE;
	}

	private static class LazyHolder {
		private static final WorkbenchPartListener INSTANCE = new WorkbenchPartListener();
	}

	private Logger logger = new Logger("WorkbenchPartListener");
	private List<ITextEditor> editors = new ArrayList<>();

	public void init() {
		try {
			logger.debug("Init WorkbenchListener.");
			IWorkbenchWindow[] windows = PlatformUI.getWorkbench().getWorkbenchWindows();
			for (IWorkbenchWindow window : windows) {
				IWorkbenchPage[] pages = window.getPages();
				for (IWorkbenchPage page : pages) {
					IEditorReference[] editorReferences = page.getEditorReferences();

					page.addPartListener(this);

					for (IEditorReference editorRef : editorReferences) {
						IEditorPart editorPart = editorRef.getEditor(false);
						if (editorPart instanceof ITextEditor textEditor) {
							IDocument document = EditorUtils.getDocument(textEditor);
							URI uri = EditorUtils.getUri(textEditor);
							getLanguageServerWrapper().connectDocument(document);
							logger.info("Connect " + uri.toString() + " to LS when init.");

							InlineCompletionTrigger.getInstance().register(textEditor);
						}
					}
				}
			}
		} catch (Exception e) {
			logger.error("Failed to init WorkbenchListener.", e);
		}
	}

	@Override
	public void partOpened(IWorkbenchPart part) {
		try {
			if (part != null) {
				logger.debug("Handle event partOpened: " + part.toString());
				ITextEditor textEditor = (ITextEditor) part.getAdapter(ITextEditor.class);
				if (textEditor != null) {
					IDocument document = EditorUtils.getDocument(textEditor);
					URI uri = EditorUtils.getUri(textEditor);
					getLanguageServerWrapper().connectDocument(document);
					logger.info("Connect " + uri.toString() + " to LS when partOpened.");

					InlineCompletionTrigger.getInstance().register(textEditor);
					editors.add(textEditor);
				}
			}
		} catch (Exception e) {
			logger.error("Failed to handle event partOpened.", e);
		}
	}

	@Override
	public void partClosed(IWorkbenchPart part) {
		try {
			if (part != null) {
				logger.debug("Handle event partClosed: " + part.toString());
				ITextEditor textEditor = (ITextEditor) part.getAdapter(ITextEditor.class);
				if (editors.contains(textEditor)) {
					URI uri = EditorUtils.getUri(textEditor);
					getLanguageServerWrapper().disconnect(uri);
					logger.info("Disconnect " + uri.toString() + " from LS.");

					InlineCompletionTrigger.getInstance().unregister(textEditor);
					editors.remove(textEditor);
				}
			}
		} catch (Exception e) {
			logger.error("Failed to handle event partClosed.", e);
		}
	}

	@Override
	public void partActivated(IWorkbenchPart part) {
	}

	@Override
	public void partDeactivated(IWorkbenchPart part) {
	}

	@Override
	public void partBroughtToTop(IWorkbenchPart part) {
	}

	private LanguageServerWrapper getLanguageServerWrapper() {
		LanguageServerWrapper server = LanguageServerService.getInstance().getServer();
		return server;
	}
}
