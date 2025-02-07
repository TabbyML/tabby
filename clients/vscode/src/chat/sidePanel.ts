import { ExtensionContext, WebviewViewProvider, WebviewView } from "vscode";
import type { ChatCommand, EditorContext } from "tabby-chat-panel";
import { ChatWebview } from "./webview";
import type { ContextVariables } from "../ContextVariables";
import type { Client } from "../lsp/client";
import { GitProvider } from "../git/GitProvider";

export class ChatSidePanelProvider implements WebviewViewProvider {
  readonly chatWebview: ChatWebview;

  constructor(
    private readonly context: ExtensionContext,
    private readonly client: Client,
    private readonly contextVariables: ContextVariables,
    private readonly gitProvider: GitProvider,
  ) {
    this.chatWebview = new ChatWebview(this.context, this.client, this.gitProvider);
  }

  async resolveWebviewView(webviewView: WebviewView) {
    this.chatWebview.init(webviewView.webview);

    this.contextVariables.chatSidePanelVisible = webviewView.visible;
    webviewView.onDidChangeVisibility(() => {
      this.contextVariables.chatSidePanelVisible = webviewView.visible;
    });

    webviewView.onDidDispose(() => {
      this.chatWebview.dispose();
    });
  }

  executeCommand(command: ChatCommand) {
    this.chatWebview.executeCommand(command);
  }

  addRelevantContext(context: EditorContext) {
    this.chatWebview.addRelevantContext(context);
  }
}
