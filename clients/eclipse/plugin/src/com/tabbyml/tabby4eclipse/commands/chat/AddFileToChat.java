package com.tabbyml.tabby4eclipse.commands.chat;

import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;

import com.tabbyml.tabby4eclipse.Logger;
import com.tabbyml.tabby4eclipse.chat.ChatView;
import com.tabbyml.tabby4eclipse.chat.ChatViewUtils;
import com.tabbyml.tabby4eclipse.editor.EditorUtils;

public class AddFileToChat extends AbstractHandler {
	private Logger logger = new Logger("Commands.Chat.AddFileToChat");

	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		logger.debug("Open chat view and send message: AddFileToChat");
		ChatView chatView = ChatViewUtils.openChatView();
		if (chatView != null) {
			chatView.addActiveEditorAsContext();
		}
		return null;
	}

	@Override
	public boolean isEnabled() {
		return EditorUtils.getActiveTextEditor() != null;
	}

}
