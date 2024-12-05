## 2.0.0

Since version 2.0, the vim-tabby plugin is designed as two parts:
1. **LSP Client Extension**:
   - Relies on an LSP client and extends it with methods (such as `textDocument/inlineCompletion`) to communicate with the tabby-agent.
   - Note: The Node.js script of tabby-agent is no longer a built-in part of the vim-tabby plugin. You need to install tabby-agent separately via npm, and the LSP client will launch it using the command `npx tabby-agent --stdio`.
2. **Inline Completion UI**:
   - Automatically triggers inline completion requests when typing.
   - Renders the inline completion text as ghost text.
   - Sets up actions with keyboard shortcuts to accept or dismiss the inline completion.
