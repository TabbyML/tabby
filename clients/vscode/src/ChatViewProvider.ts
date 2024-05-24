import { ExtensionContext, WebviewViewProvider, WebviewView, workspace, Uri, env } from "vscode";
import { ServerApi, ChatMessage, Context } from "tabby-chat-panel";

import { agent } from "./agent";
import { createClient } from "./chatPanel";

export class ChatViewProvider implements WebviewViewProvider {
  webview?: WebviewView;
  client?: ServerApi;
  private pendingMessages: ChatMessage[] = [];

  constructor(private readonly context: ExtensionContext) {}

  public async resolveWebviewView(webviewView: WebviewView) {
    this.webview = webviewView;
    const extensionUri = this.context.extensionUri;
    const { server } = agent().getConfig();

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [extensionUri],
    };
    webviewView.webview.html = await this._getWebviewContent();

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
        this.pendingMessages.forEach((message) => this.client?.sendMessage(message));
        this.client?.init({
          fetcherOptions: {
            authorization: server.token,
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

  private async _getWebviewContent() {
    const { server } = agent().getConfig();
    return `
      <!DOCTYPE html>
      <html lang="en">
        <head>
          <meta charset="UTF-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <title>Tabby</title>     
        </head>
        <style>
          html, body {
            background: transparent;
          }
          html, body, iframe {
            padding: 0;
            margin: 0;
            box-sizing: border-box;
            overflow: hidden;
          }
          iframe {
            border-width: 0;
            width: 100%;
            height: 100vh;
          }
        </style>
        <body>
          <script>const vscode = acquireVsCodeApi();</script>
          <script defer>
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

              syncTheme();
            }
          </script>
          <script>
            function iframeLoaded () {
              vscode.postMessage({ action: 'rendered' });
            }
          </script>
          <iframe
            id="chat"
            src="${server.endpoint}/chat?max-width=5xl"
            onload="iframeLoaded(this)" />
        </body>
      </html>
    `;
  }

  public getWebview() {
    return this.webview;
  }

  public sendMessage(message: ChatMessage) {
    if (!this.client) {
      this.pendingMessages.push(message);
    } else {
      this.client.sendMessage(message);
    }
  }
}
