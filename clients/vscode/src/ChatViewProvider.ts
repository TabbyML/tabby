import { ExtensionContext, Disposable, Webview, WebviewPanel, window, Uri, ViewColumn, WebviewViewProvider, WebviewView, TextDocument } from "vscode";
import { getUri } from "./utils";

import { createAgentInstance, disposeAgentInstance } from "./agent";

export class ChatViewProvider implements WebviewViewProvider {
  webview?: WebviewView;

  constructor(private readonly context: ExtensionContext) {}

  public async resolveWebviewView(webviewView: WebviewView) {
    this.webview = webviewView;
    const extensionUri = this.context.extensionUri

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [extensionUri],
    };
    webviewView.webview.html = await this._getWebviewContent(webviewView.webview, extensionUri);
  }

  private async _getWebviewContent(webview: Webview, extensionUri: Uri) {
    const agent = await createAgentInstance(this.context);
    const { server } = agent.getConfig()
    const scriptUri = getUri(webview, extensionUri, ['chat-panel', "index.js"]);
    return `
      <!DOCTYPE html>
      <html lang="en">
        <head>
          <meta charset="UTF-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <title>Tabby</title>     
        </head>
        <style>
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
          <script>window.endpoint="${server.endpoint}"</script>
          <script>window.token="${server.token}"</script>
          <script defer>
            window.onload = function () {
              const vscode = acquireVsCodeApi();
              const chatIframe = document.getElementById("chat");
            
              window.addEventListener("message", (event) => {
                console.log('window.addEventListener', event.data);
                if (event.data) {
                  if (event.data.data) {
                    chatIframe.contentWindow.postMessage(event.data.data[0], "http://localhost:8080");
                  } else {
                    console.log('data from iframe', event.data);
                    vscode.postMessage(event.data);
                  }
                }
              });
            }
          </script>
          <iframe id="chat" src="http://localhost:8080/chat" />
        </body>
      </html>
    `;
  }

  public getWebview () {
    return this.webview
  }
}