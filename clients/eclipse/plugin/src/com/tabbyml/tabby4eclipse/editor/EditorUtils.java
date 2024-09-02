package com.tabbyml.tabby4eclipse.editor;

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
}
