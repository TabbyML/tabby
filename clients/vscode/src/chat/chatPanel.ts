import { window, ExtensionContext, ViewColumn } from "vscode";
import { v4 as uuid } from "uuid";
import { ChatWebview } from "./webview";
import type { Client } from "../lsp/Client";
import type { GitProvider } from "../git/GitProvider";

export async function createChatPanel(context: ExtensionContext, client: Client, gitProvider: GitProvider) {
  const id = uuid();
  const panel = window.createWebviewPanel(`tabby.chat.panel-${id}`, "Tabby", ViewColumn.One, {
    retainContextWhenHidden: true,
  });

  const chatWebview = new ChatWebview(context, client, gitProvider);
  chatWebview.init(panel.webview);

  panel.onDidDispose(() => {
    chatWebview.dispose();
  });
}
