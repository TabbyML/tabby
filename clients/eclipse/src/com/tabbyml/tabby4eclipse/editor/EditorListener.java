package com.tabbyml.tabby4eclipse.editor;

import org.eclipse.jface.text.IDocument;
import org.eclipse.jface.text.ITextViewer;
import org.eclipse.lsp4e.LSPEclipseUtils;
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
import com.tabbyml.tabby4eclipse.lsp.LanguageServerService;

public class EditorListener implements IPartListener {
	public static EditorListener getInstance() {
		return LazyHolder.INSTANCE;
	}

	private static class LazyHolder {
		private static final EditorListener INSTANCE = new EditorListener();
	}

	private Logger logger = new Logger("EditorListener");

	public void init() {
		try {
			IWorkbenchWindow[] windows = PlatformUI.getWorkbench().getWorkbenchWindows();
			for (IWorkbenchWindow window : windows) {
				IWorkbenchPage[] pages = window.getPages();
				for (IWorkbenchPage page : pages) {
					IEditorReference[] editorReferences = page.getEditorReferences();

					page.addPartListener(this);

					for (IEditorReference editorRef : editorReferences) {
						IEditorPart editorPart = editorRef.getEditor(false);
						if (editorPart instanceof ITextEditor) {
							IDocument document = LSPEclipseUtils.getDocument(editorPart.getEditorInput());
							getLanguageServerWrapper().connectDocument(document);
							logger.info(
									"Connect document: " + LSPEclipseUtils.toUri(document) + " to server when init.");
							
							getInlineCompletionService().register((ITextEditor) editorPart);
						}
					}
				}
			}
		} catch (Exception e) {
			logger.error("Error when initializing editor manager.", e);
		}
	}

	@Override
	public void partOpened(IWorkbenchPart part) {
		try {
			if (part != null) {
				IEditorPart editorPart = (IEditorPart) part.getAdapter(IEditorPart.class);
				if (editorPart instanceof ITextEditor) {
					IDocument document = LSPEclipseUtils.getDocument(editorPart.getEditorInput());
					getLanguageServerWrapper().connectDocument(document);
					logger.info("Connect document: " + LSPEclipseUtils.toUri(document) + " to server when opened.");
					
					getInlineCompletionService().register((ITextEditor) editorPart);
				}
			}
		} catch (Exception e) {
			logger.error("Error when part opened.", e);
		}
	}

	@Override
	public void partClosed(IWorkbenchPart part) {
		try {
			if (part != null) {
				IEditorPart editorPart = (IEditorPart) part.getAdapter(IEditorPart.class);
				if (editorPart instanceof ITextEditor) {
					IDocument document = LSPEclipseUtils.getDocument(editorPart.getEditorInput());
					getLanguageServerWrapper().disconnect(LSPEclipseUtils.toUri(document));
					logger.info("Disconnect document: " + LSPEclipseUtils.toUri(document));
					
					getInlineCompletionService().unregister((ITextEditor) editorPart);
				}
			}
		} catch (Exception e) {
			logger.error("Error when part closed.", e);
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
		if (server == null) {
			logger.error("Cannot initialize EditorManager. Language server is not available.");
		}
		return server;
	}
	
	private InlineCompletionService getInlineCompletionService() {
		return InlineCompletionService.getInstance();
	}
}
