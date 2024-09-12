import {
  workspace,
  Uri,
  env,
  LogOutputChannel,
  window,
  Position,
  Range,
  Selection,
  TextEditorRevealType,
  ViewColumn,
  WorkspaceFolder,
  TextDocument,
  commands,
  WebviewPanel,
} from "vscode";
import type { ChatMessage, Context, NavigateOpts } from "tabby-chat-panel";
import { createClient } from "./chatPanel";
import { BaseChatView } from "./BaseChatView";

// TODO(zhizhg): abstruct a base class with ChatViewProvider
export class ChatPanelViewProvider extends BaseChatView {
  webview?: WebviewPanel;

  // The method is called when the chat panel first opened
  public async resolveWebviewView(webviewView: WebviewPanel) {
    this.webview = webviewView;
    const extensionUri = this.context.extensionUri;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [extensionUri],
    };

    this.client = createClient(webviewView, {
      navigate: async (context: Context, opts?: NavigateOpts) => {
        if (opts?.openInEditor) {
          const document = await resolveDocument(this.logger, workspace.workspaceFolders, context.filepath);
          if (!document) {
            throw new Error(`File not found: ${context.filepath}`);
          }

          const newEditor = await window.showTextDocument(document, {
            viewColumn: ViewColumn.Active,
            preview: false,
            preserveFocus: true,
          });

          // Move the cursor to the specified line
          const start = new Position(Math.max(0, context.range.start - 1), 0);
          const end = new Position(context.range.end, 0);
          newEditor.selection = new Selection(start, end);
          newEditor.revealRange(new Range(start, end), TextEditorRevealType.InCenter);

          return;
        }

        if (context?.filepath && context?.git_url) {
          const serverInfo = await this.agent.fetchServerInfo();

          const url = new URL(`${serverInfo.config.endpoint}/files`);
          const searchParams = new URLSearchParams();

          searchParams.append("redirect_filepath", context.filepath);
          searchParams.append("redirect_git_url", context.git_url);
          url.search = searchParams.toString();

          const lineHash = this.formatLineHashForCodeBrowser(context.range);
          if (lineHash) {
            url.hash = lineHash;
          }

          await env.openExternal(Uri.parse(url.toString()));
        }
      },
      refresh: async () => {
        const serverInfo = await this.agent.fetchServerInfo();
        await this.displayChatPage(serverInfo.config.endpoint, { force: true });
        return;
      },
      onSubmitMessage: async (msg: string, relevantContext?: Context[]) => {
        const editor = window.activeTextEditor;
        const chatMessage: ChatMessage = {
          message: msg,
          relevantContext: [],
        };
        if (editor) {
          const fileContext = ChatPanelViewProvider.getFileContextFromSelection({
            editor,
            gitProvider: this.gitProvider,
          });
          if (fileContext)
            // active selection
            chatMessage.activeContext = fileContext;
        }
        if (relevantContext) {
          chatMessage.relevantContext = chatMessage.relevantContext?.concat(relevantContext);
        }

        // FIXME: maybe deduplicate on chatMessage.relevantContext
        this.sendMessage(chatMessage);
      },
      onApplyInEditor: (content: string) => {
        const editor = window.activeTextEditor;
        if (editor) {
          const document = editor.document;
          const selection = editor.selection;

          // Determine the indentation for the content
          // The calculation is based solely on the indentation of the first line
          const lineText = document.lineAt(selection.start.line).text;
          const match = lineText.match(/^(\s*)/);
          const indent = match ? match[0] : "";

          // Determine the indentation for the content's first line
          // Note:
          // If using spaces, selection.start.character = 1 means 1 space
          // If using tabs, selection.start.character = 1 means 1 tab
          const indentUnit = indent[0];
          const indentAmountForTheFirstLine = Math.max(indent.length - selection.start.character, 0);
          const indentForTheFirstLine = indentUnit?.repeat(indentAmountForTheFirstLine) || "";

          // Indent the content
          const indentedContent = indentForTheFirstLine + content.replaceAll("\n", "\n" + indent);

          // Apply into the editor
          editor.edit((editBuilder) => {
            editBuilder.replace(selection, indentedContent);
          });
        }
      },
    });

    // At this point, if the server instance is not set up, agent.status is 'notInitialized'.
    // We check for the presence of the server instance by verifying serverInfo.health["webserver"].
    const serverInfo = await this.agent.fetchServerInfo();

    if (serverInfo.health && serverInfo.health["webserver"]) {
      const serverInfo = await this.agent.fetchServerInfo();
      this.displayChatPage(serverInfo.config.endpoint);
    } else {
      this.displayDisconnectedPage();
    }

    this.agent.on("didChangeStatus", async (status) => {
      if (status !== "disconnected") {
        const serverInfo = await this.agent.fetchServerInfo();
        this.displayChatPage(serverInfo.config.endpoint);
        this.refreshChatPage();
      } else if (this.isChatPageDisplayed) {
        this.displayDisconnectedPage();
      }
    });

    this.agent.on("didUpdateServerInfo", async () => {
      const serverInfo = await this.agent.fetchServerInfo();
      this.displayChatPage(serverInfo.config.endpoint, { force: true });
      this.refreshChatPage();
    });

    // The event will not be triggered during the initial rendering.
    webviewView.onDidChangeViewState(() => {
      if (webviewView.visible) {
        this.refreshChatPage();
      }

      commands.executeCommand("setContext", "tabby.chatViewVisible", webviewView.visible);
    });

    webviewView.webview.onDidReceiveMessage((message) => {
      switch (message.action) {
        case "rendered": {
          setTimeout(() => {
            this.refreshChatPage();
          }, 300);
          return;
        }
        case "copy": {
          env.clipboard.writeText(message.data);
          return;
        }
      }
    });

    workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("workbench.colorTheme")) {
        this.webview?.webview.postMessage({ action: "sync-theme" });
      }
    });
  }

  public getWebview() {
    return this.webview;
  }

  public sendMessage(message: ChatMessage) {
    if (!this.client) {
      this.pendingMessages.push(message);
    } else {
      this.sendMessageToChatPanel(message);
    }
  }
}

async function resolveDocument(
  logger: LogOutputChannel,
  folders: readonly WorkspaceFolder[] | undefined,
  filepath: string,
): Promise<TextDocument | null> {
  if (filepath.startsWith("file://")) {
    const absoluteFilepath = Uri.parse(filepath, true);
    return workspace.openTextDocument(absoluteFilepath);
  }

  if (!folders) {
    return null;
  }

  for (const root of folders) {
    const absoluteFilepath = Uri.joinPath(root.uri, filepath);
    try {
      return await workspace.openTextDocument(absoluteFilepath);
    } catch (err) {
      // Do nothing, file doesn't exists.
    }
  }

  logger.info("File not found in workspace folders, trying with findFiles...");

  const files = await workspace.findFiles(filepath, undefined, 1);
  if (files[0]) {
    return workspace.openTextDocument(files[0]);
  }

  return null;
}
