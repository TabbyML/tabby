import { ExtensionContext, TextEditor, window, WebviewPanel } from "vscode";
import type { ServerApi, ChatMessage, Context } from "tabby-chat-panel";
import { WebviewHelper } from "./WebviewHelper";
import type { AgentFeature as Agent } from "../lsp/AgentFeature";
import { GitProvider } from "../git/GitProvider";
import { getLogger } from "../logger";

export class ChatPanelViewProvider {
  webview?: WebviewPanel;
  client?: ServerApi;
  private webviewHelper: WebviewHelper;

  constructor(
    private readonly context: ExtensionContext,
    agent: Agent,
    gitProvider: GitProvider,
  ) {
    const logger = getLogger();
    this.webviewHelper = new WebviewHelper(context, agent, logger, gitProvider);
  }

  static getFileContextFromSelection({
    editor,
    gitProvider,
  }: {
    editor: TextEditor;
    gitProvider: GitProvider;
  }): Context | null {
    return WebviewHelper.getFileContextFromSelection({ editor, gitProvider });
  }

  // The method is called when the chat panel first opened
  public async resolveWebviewView(webviewView: WebviewPanel) {
    this.webview = webviewView;
    const extensionUri = this.context.extensionUri;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [extensionUri],
    };
    this.webviewHelper.setWebview(webviewView.webview);

    this.client = this.webviewHelper.createChatClient(webviewView.webview);
    this.webviewHelper.setClient(this.client);

    await this.webviewHelper.displayPageBasedOnServerStatus();
    this.webviewHelper.addAgentEventListeners();

    // The event will not be triggered during the initial rendering.
    webviewView.onDidChangeViewState(() => {
      if (webviewView.visible) {
        this.webviewHelper.refreshChatPage();
      }
    });

    webviewView.webview.onDidReceiveMessage((message) => {
      switch (message.action) {
        case "sync-theme": {
          this.client?.updateTheme(message.style, WebviewHelper.getColorThemeString(window.activeColorTheme.kind));
          return;
        }
      }
    });
  }

  public getWebview() {
    return this.webview;
  }

  public sendMessage(message: ChatMessage) {
    this.webviewHelper.sendMessage(message);
  }

  public addRelevantContext(context: Context) {
    this.webviewHelper.addRelevantContext(context);
  }
}
