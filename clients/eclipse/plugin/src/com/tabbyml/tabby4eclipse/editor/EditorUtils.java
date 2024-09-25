package com.tabbyml.tabby4eclipse.editor;

import java.net.URI;

import org.eclipse.core.resources.IFile;
import org.eclipse.jface.text.IDocument;
import org.eclipse.jface.text.IDocumentExtension4;
import org.eclipse.jface.text.ITextSelection;
import org.eclipse.jface.text.ITextViewer;
import org.eclipse.jface.text.ITextViewerExtension5;
import org.eclipse.jface.viewers.ISelection;
import org.eclipse.lsp4e.LSPEclipseUtils;
import org.eclipse.swt.SwtCallable;
import org.eclipse.swt.custom.StyledText;
import org.eclipse.swt.widgets.Display;
import org.eclipse.ui.IEditorPart;
import org.eclipse.ui.IWorkbenchPage;
import org.eclipse.ui.IWorkbenchWindow;
import org.eclipse.ui.PlatformUI;
import org.eclipse.ui.texteditor.ITextEditor;

public class EditorUtils {
	public static IWorkbenchPage getActiveWorkbenchPage() {
		IWorkbenchWindow window = PlatformUI.getWorkbench().getActiveWorkbenchWindow();
		if (window != null) {
			IWorkbenchPage page = window.getActivePage();
			return page;
		}
		return null;
	}

	public static ITextEditor getActiveTextEditor() {
		IWorkbenchPage page = getActiveWorkbenchPage();
		if (page != null) {
			IEditorPart activeEditor = page.getActiveEditor();
			if (activeEditor instanceof ITextEditor textEditor) {
				return textEditor;
			}
		}
		return null;
	}

	public static boolean isActiveEditor(ITextEditor textEditor) {
		return textEditor == EditorUtils.getActiveTextEditor();
	}

	public static ITextViewer getTextViewer(ITextEditor textEditor) {
		return (ITextViewer) textEditor.getAdapter(ITextViewer.class);
	}

	public static IDocument getDocument(ITextEditor textEditor) {
		return LSPEclipseUtils.getDocument(textEditor.getEditorInput());
	}

	public static URI getUri(ITextEditor textEditor) {
		IDocument document = getDocument(textEditor);
		return LSPEclipseUtils.toUri(document);
	}

	public static StyledText getStyledTextWidget(ITextEditor textEditor) {
		return getTextViewer(textEditor).getTextWidget();
	}

	public static Display getDisplay(ITextEditor textEditor) {
		return getStyledTextWidget(textEditor).getDisplay();
	}

	public static void asyncExec(Runnable runnable) {
		PlatformUI.getWorkbench().getDisplay().asyncExec(runnable);
	}

	public static void asyncExec(ITextEditor textEditor, Runnable runnable) {
		getDisplay(textEditor).asyncExec(runnable);
	}

	public static void syncExec(Runnable runnable) {
		PlatformUI.getWorkbench().getDisplay().syncExec(runnable);
	}

	public static void syncExec(ITextEditor textEditor, Runnable runnable) {
		getDisplay(textEditor).syncExec(runnable);
	}

	public static <T, E extends Exception> T syncCall(SwtCallable<T, E> callable) throws E {
		return PlatformUI.getWorkbench().getDisplay().syncCall(callable);
	}

	public static <T, E extends Exception> T syncCall(ITextEditor textEditor, SwtCallable<T, E> callable) throws E {
		return getDisplay(textEditor).syncCall(callable);
	}

	public static long getDocumentModificationStamp(ITextEditor textEditor) {
		IDocument document = getDocument(textEditor);
		if (document instanceof IDocumentExtension4 documentExt) {
			return documentExt.getModificationStamp();
		} else if (document != null) {
			IFile file = LSPEclipseUtils.getFile(document);
			if (file != null) {
				return file.getModificationStamp();
			}
		}
		return IDocumentExtension4.UNKNOWN_MODIFICATION_STAMP;
	}

	public static int getCurrentOffsetInDocument(ITextEditor textEditor) throws IllegalStateException {
		return syncCall(textEditor, () -> {
			int offset = -1;
			ISelection selection = textEditor.getSelectionProvider().getSelection();
			if (selection instanceof ITextSelection textSelection) {
				offset = textSelection.getOffset();
			}
			if (offset != -1) {
				return offset;
			}

			int offsetInWidget = getStyledTextWidget(textEditor).getCaretOffset();
			ITextViewer textViewer = getTextViewer(textEditor);
			if (textViewer instanceof ITextViewerExtension5 textViewerExt) {
				offset = textViewerExt.widgetOffset2ModelOffset(offsetInWidget);
			}
			if (offset != -1) {
				return offset;
			}

			throw new IllegalStateException("Failed to get current offset in document.");
		});
	}
}
