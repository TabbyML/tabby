import { ExtensionContext, WebviewViewProvider, WebviewView } from "vscode";
import { ServerApi, ChatMessage } from 'tabby-chat-panel'

import { createAgentInstance, disposeAgentInstance } from "./agent";
import { createClient } from "./vscode";

export class ChatViewProvider implements WebviewViewProvider {
  webview?: WebviewView;
  client?: ServerApi;
  private pendingMessages: ChatMessage[] = [];

  constructor(private readonly context: ExtensionContext) {}

  public async resolveWebviewView(webviewView: WebviewView) {
    this.webview = webviewView;
    const extensionUri = this.context.extensionUri

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [extensionUri],
    };
    webviewView.webview.html = await this._getWebviewContent();

    this.client = createClient(webviewView, {
      navigate: () => {
        console.log('TODO')
      }
    })

    webviewView.webview.onDidReceiveMessage(message => {
      if (message.action === 'rendered') {
        this.pendingMessages.forEach(message => this.client?.sendMessage(message))
        this.client?.init({
          fetcherOptions: {
            authorization: "auth_fa450615a8cd4e77a35cd9fa61e5008f"
          }
        })
      }
    });
  }

  private async _getWebviewContent() {
    const agent = await createAgentInstance(this.context);
    const { server } = agent.getConfig()
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
          <script>const vscode = acquireVsCodeApi();</script>
          <script defer>
            window.onload = function () {
              const chatIframe = document.getElementById("chat");
              window.addEventListener("message", (event) => {
                console.log('window.addEventListener', event.data);
                if (event.data) {
                  if (event.data.data) {
                    chatIframe.contentWindow.postMessage(event.data.data[0], "${server.endpoint}");
                  } else {
                    console.log('data from iframe', event.data);
                    vscode.postMessage(event.data);
                  }
                }
              });
            }
          </script>
          <script>
            function iframeLoaded () {
              vscode.postMessage({ action: 'rendered' });
            }
          </script>
          <iframe
            id="chat"
            src="${server.endpoint}/chat"
            onload="iframeLoaded(this)" />
        </body>
      </html>
    `;
  }

  public getWebview () {
    return this.webview
  }

  public sendMessage (message: ChatMessage) {
    if (!this.client) {
      this.pendingMessages.push(message)
    } else {
      this.client.sendMessage(message)
    }
  }
}