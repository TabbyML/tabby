package com.tabbyml.tabby4eclipse.commands.chat;

import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;
import org.eclipse.ui.IEditorPart;
import org.eclipse.ui.IWorkbenchPage;
import org.eclipse.ui.IWorkbenchPart;
import org.eclipse.ui.PartInitException;

import com.tabbyml.tabby4eclipse.DebouncedRunnable;
import com.tabbyml.tabby4eclipse.Logger;
import com.tabbyml.tabby4eclipse.chat.ChatView;
import com.tabbyml.tabby4eclipse.chat.ChatViewUtils;
import com.tabbyml.tabby4eclipse.editor.EditorUtils;

public class ToggleChatView extends AbstractHandler {
	private Logger logger = new Logger("Commands.Chat.ToggleChatView");
	private static final int DEBOUNCE_INTERVAL = 50; // ms

	// prevent toggle command called many times within a short time
	private DebouncedRunnable runnable = new DebouncedRunnable(() -> {
		EditorUtils.syncExec(() -> {
			logger.debug("Toggle chat view.");
			IWorkbenchPage page = EditorUtils.getActiveWorkbenchPage();
			if (page != null) {
				boolean chatPanelFocused = page.getActivePart() == ChatViewUtils.findOpenedView();
				if (chatPanelFocused) {
					logger.debug("Switch to Editor.");
					IEditorPart editorPart = page.getActiveEditor();
					if (editorPart != null) {
						page.activate(editorPart);
					}
				} else {
					logger.debug("Switch to ChatView.");
					ChatView chatView = ChatViewUtils.openChatView();
					if (chatView != null) {
						page.activate(chatView);

						String selectedText = EditorUtils.getSelectedText();
						if (selectedText != null && !selectedText.isBlank()) {
							logger.debug("Send message: AddSelectionToChat");
							chatView.addSelectedTextAsContext();
						}
					}
				}
			}
		});
	}, DEBOUNCE_INTERVAL);

	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		runnable.call();
		return null;
	}

	@Override
	public boolean isEnabled() {
		return true;
	}

}
