import { ExtensionContext, WebviewViewProvider, WebviewView, window } from "vscode";
import type { ServerApi, ChatMessage, Context } from "tabby-chat-panel";
import { WebviewHelper } from "./WebviewHelper";
import { Client } from "../lsp/Client";
import type { LogOutputChannel } from "../logger";
import { GitProvider } from "../git/GitProvider";

export class ChatSideViewProvider implements WebviewViewProvider {
  webview?: WebviewView;
  client?: ServerApi;
  private webviewHelper: WebviewHelper;

  constructor(
    private readonly context: ExtensionContext,
    client: Client,
    logger: LogOutputChannel,
    gitProvider: GitProvider,
  ) {
    this.webviewHelper = new WebviewHelper(context, client, logger, gitProvider);
  }

  // The method is called when the chat panel first opened
  public async resolveWebviewView(webviewView: WebviewView) {
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

    this.webviewHelper.syncActiveSelection(window.activeTextEditor);
    this.webviewHelper.addTextEditorEventListeners();

    // The event will not be triggered during the initial rendering.
    webviewView.onDidChangeVisibility(() => {
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
