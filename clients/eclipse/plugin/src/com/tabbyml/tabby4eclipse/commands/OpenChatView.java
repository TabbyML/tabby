package com.tabbyml.tabby4eclipse.commands;

import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;

import com.tabbyml.tabby4eclipse.chat.ChatView;

public class OpenChatView extends AbstractHandler {

	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		ChatView.openChatView();
		return null;
	}

	@Override
	public boolean isEnabled() {
		return true;
	}

}
