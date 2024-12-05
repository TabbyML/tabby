package com.tabbyml.tabby4eclipse.commands.chat;

import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;

import com.tabbyml.tabby4eclipse.Logger;
import com.tabbyml.tabby4eclipse.chat.ChatViewUtils;

public class OpenChatView extends AbstractHandler {
	private Logger logger = new Logger("Commands.Chat.OpenChatView");

	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		logger.debug("Open chat view.");
		ChatViewUtils.openChatView();
		return null;
	}

	@Override
	public boolean isEnabled() {
		return true;
	}

}
