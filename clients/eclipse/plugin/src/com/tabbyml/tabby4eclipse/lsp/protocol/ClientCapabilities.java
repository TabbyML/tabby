package com.tabbyml.tabby4eclipse.lsp.protocol;

public class ClientCapabilities {
	private TextDocumentClientCapabilities textDocument;
	private TabbyClientCapabilities tabby;

	public ClientCapabilities() {
	}

	public TextDocumentClientCapabilities getTextDocument() {
		return textDocument;
	}

	public void setTextDocument(TextDocumentClientCapabilities textDocument) {
		this.textDocument = textDocument;
	}

	public TabbyClientCapabilities getTabby() {
		return tabby;
	}

	public void setTabby(TabbyClientCapabilities tabby) {
		this.tabby = tabby;
	}

	public static class TextDocumentClientCapabilities {
		private boolean completion;
		private boolean inlineCompletion;

		public TextDocumentClientCapabilities() {
			this.completion = false;
			this.inlineCompletion = false;
		}

		public boolean getCompletion() {
			return completion;
		}

		public void setCompletion(boolean completion) {
			this.completion = completion;
		}

		public boolean getInlineCompletion() {
			return inlineCompletion;
		}

		public void setInlineCompletion(boolean inlineCompletion) {
			this.inlineCompletion = inlineCompletion;
		}
	}

	public static class TabbyClientCapabilities {
		private boolean agent;
		private boolean workspaceFileSystem;
		private boolean dataStore;
		private boolean languageSupport;
		private boolean gitProvider;
		private boolean editorOptions;

		public TabbyClientCapabilities() {
			this.agent = false;
			this.workspaceFileSystem = false;
			this.dataStore = false;
			this.languageSupport = false;
			this.gitProvider = false;
			this.editorOptions = false;
		}

		public boolean getAgent() {
			return agent;
		}

		public void setAgent(boolean agent) {
			this.agent = agent;
		}

		public boolean getWorkspaceFileSystem() {
			return workspaceFileSystem;
		}

		public void setWorkspaceFileSystem(boolean workspaceFileSystem) {
			this.workspaceFileSystem = workspaceFileSystem;
		}

		public boolean getDataStore() {
			return dataStore;
		}

		public void setDateStore(boolean dataStore) {
			this.dataStore = dataStore;
		}

		public boolean getLanguageSupport() {
			return languageSupport;
		}

		public void setLanguageSupport(boolean languageSupport) {
			this.languageSupport = languageSupport;
		}

		public boolean getGitProvider() {
			return gitProvider;
		}

		public void setGitProvider(boolean gitProvider) {
			this.gitProvider = gitProvider;
		}

		public boolean getEditorOptions() {
			return editorOptions;
		}

		public void setEditorOptions(boolean editorOptions) {
			this.editorOptions = editorOptions;
		}
	}
}
