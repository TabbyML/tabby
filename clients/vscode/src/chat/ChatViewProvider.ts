import { ExtensionContext, WebviewViewProvider, WebviewView, workspace, Uri, env } from "vscode";
import type { ServerApi, ChatMessage, Context } from "tabby-chat-panel";
import { ServerConfig } from "tabby-agent";
import hashObject from "object-hash";
import { Config } from "../Config";
import { createClient } from "./chatPanel";

// FIXME(wwayne): Example code has webview removed case, not sure when it would happen, need to double check
export class ChatViewProvider implements WebviewViewProvider {
  webview?: WebviewView;
  client?: ServerApi;
  private pendingMessages: ChatMessage[] = [];

  constructor(
    private readonly context: ExtensionContext,
    private readonly config: Config,
  ) {}

  public async resolveWebviewView(webviewView: WebviewView) {
    this.webview = webviewView;
    const extensionUri = this.context.extensionUri;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [extensionUri],
    };
    webviewView.webview.html = this._getWebviewContent(this.config.server);
    this.config.on("updatedServerConfig", () => {
      webviewView.webview.html = this._getWebviewContent(this.config.server);
    });

    this.client = createClient(webviewView, {
      navigate: async (context: Context) => {
        if (context?.filepath && context?.git_url) {
          const url = `${context.git_url}/blob/main/${context.filepath}#L${context.range.start}-L${context.range.end}`;
          await env.openExternal(Uri.parse(url));
        }
      },
    });

    webviewView.webview.onDidReceiveMessage((message) => {
      if (message.action === "rendered") {
        this.webview?.webview.postMessage({ action: "sync-theme" });
        this.pendingMessages.forEach((message) => this.client?.sendMessage(message));
        this.client?.init({
          fetcherOptions: {
            authorization: this.config.server.token,
          },
        });
      }
    });

    workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("workbench.colorTheme")) {
        this.webview?.webview.postMessage({ action: "sync-theme" });
      }
    });
  }

  private _getWebviewContent(server: ServerConfig) {
    const styleUri = this.webview?.webview.asWebviewUri(
      Uri.joinPath(this.context.extensionUri, "assets", "chat-panel.css"),
    );
    return `
      <!DOCTYPE html>
      <html lang="en">
        <!--hash: ${hashObject(server)}-->
        <head>
          <meta charset="UTF-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <title>Tabby</title>
          <link rel="preconnect" href="${server.endpoint}">
          <link href="${styleUri}" rel="stylesheet">
          <script defer>
            const vscode = acquireVsCodeApi();
          
            function iframeLoaded () {
              vscode.postMessage({ action: 'rendered' });
            }

            window.onload = function () {
              const chatIframe = document.getElementById("chat");
        
              const syncTheme = () => {
                const parentHtmlStyle = document.documentElement.getAttribute('style');
                chatIframe.contentWindow.postMessage({ style: parentHtmlStyle }, "${server.endpoint}");
            
                let themeClass = document.body.className === 'vscode-dark' ? 'dark' : 'light'
                themeClass += ' vscode'
                chatIframe.contentWindow.postMessage({ themeClass: themeClass }, "${server.endpoint}");
              }
        
              window.addEventListener("message", (event) => {
                if (event.data) {
                  if (event.data.action === 'sync-theme') {
                    syncTheme();
                    return;
                  }

                  if (event.data.data) {
                    chatIframe.contentWindow.postMessage(event.data.data[0], "${server.endpoint}");
                  } else {
                    vscode.postMessage(event.data);
                  }
                }
              });
            }
          </script>
        </head>
        <body>
          <iframe
            id="chat"
            src="${server.endpoint}/chat?from=vscode"
            onload="iframeLoaded(this)" />
        </body>
      </html>
    `;
  }

  public getWebview() {
    return this.webview;
  }

  public sendMessage(message: ChatMessage) {
    console.log("this.client", this.client);
    if (!this.client) {
      this.pendingMessages.push(message);
    } else {
      this.client.sendMessage(message);
    }
  }
}
