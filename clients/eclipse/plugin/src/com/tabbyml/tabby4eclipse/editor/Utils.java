package com.tabbyml.tabby4eclipse.editor;

import org.eclipse.ui.IEditorPart;
import org.eclipse.ui.IWorkbenchPage;
import org.eclipse.ui.IWorkbenchWindow;
import org.eclipse.ui.PlatformUI;
import org.eclipse.ui.texteditor.ITextEditor;

public class Utils {
	public static ITextEditor getActiveTextEditor() {
		IWorkbenchWindow window = PlatformUI.getWorkbench().getActiveWorkbenchWindow();
		if (window != null) {
			IWorkbenchPage page = window.getActivePage();
			if (page != null) {
				IEditorPart activeEditor = page.getActiveEditor();
				if (activeEditor instanceof ITextEditor textEditor) {
					return textEditor;
				}
			}
		}
		return null;
	}
}
