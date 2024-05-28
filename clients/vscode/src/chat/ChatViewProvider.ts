import { ExtensionContext, WebviewViewProvider, WebviewView, workspace, Uri, env } from "vscode";
import type { ServerApi, ChatMessage, Context } from "tabby-chat-panel";
import hashObject from "object-hash";
import * as semver from "semver";
import type { ServerInfo } from "tabby-agent";
import type { AgentFeature as Agent } from "../lsp/AgentFeature";
import { createClient } from "./chatPanel";

// FIXME(wwayne): Example code has webview removed case, not sure when it would happen, need to double check
export class ChatViewProvider implements WebviewViewProvider {
  webview?: WebviewView;
  client?: ServerApi;
  private pendingMessages: ChatMessage[] = [];

  constructor(
    private readonly context: ExtensionContext,
    private readonly agent: Agent,
  ) {}

  public async resolveWebviewView(webviewView: WebviewView) {
    this.webview = webviewView;
    const extensionUri = this.context.extensionUri;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [extensionUri],
    };
    // FIXME: we need to wait for the server to be ready, consider rendering a loading indicator
    if (this.agent.status !== "ready") {
      await new Promise<void>((resolve) => {
        this.agent.on("didChangeStatus", (status) => {
          if (status === "ready") {
            resolve();
          }
        });
      });
    }
    const serverInfo = await this.agent.fetchServerInfo();
    webviewView.webview.html = this.getWebviewContent(serverInfo);
    this.agent.on("didUpdateServerInfo", (serverInfo: ServerInfo) => {
      webviewView.webview.html = this.getWebviewContent(serverInfo);
    });

    this.client = createClient(webviewView, {
      navigate: async (context: Context) => {
        if (context?.filepath && context?.git_url) {
          const url = `${context.git_url}/blob/main/${context.filepath}#L${context.range.start}-L${context.range.end}`;
          await env.openExternal(Uri.parse(url));
        }
      },
    });

    webviewView.webview.onDidReceiveMessage(async (message) => {
      if (message.action === "rendered") {
        this.webview?.webview.postMessage({ action: "sync-theme" });
        this.pendingMessages.forEach((message) => this.client?.sendMessage(message));
        const serverInfo = await this.agent.fetchServerInfo();
        if (serverInfo.config.token) {
          this.client?.init({
            fetcherOptions: {
              authorization: serverInfo.config.token,
            },
          });
        }
      }
    });

    workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("workbench.colorTheme")) {
        this.webview?.webview.postMessage({ action: "sync-theme" });
      }
    });
  }

  private isChatPanelAvailable(serverInfo: ServerInfo): boolean {
    if (!serverInfo.health || !serverInfo.health["webserver"] || !serverInfo.health["chat_model"]) {
      return false;
    }
    if (serverInfo.health["version"]) {
      let version: semver.SemVer | undefined | null = undefined;
      if (typeof serverInfo.health["version"] === "string") {
        version = semver.coerce(serverInfo.health["version"]);
      } else if (
        typeof serverInfo.health["version"] === "object" &&
        "git_describe" in serverInfo.health["version"] &&
        typeof serverInfo.health["version"]["git_describe"] === "string"
      ) {
        version = semver.coerce(serverInfo.health["version"]["git_describe"]);
      }
      if (version && semver.lt(version, "0.12.0")) {
        return false;
      }
    }
    return true;
  }

  private getWebviewContent(serverInfo: ServerInfo) {
    if (!this.isChatPanelAvailable(serverInfo)) {
      return `
        <!DOCTYPE html>
        <html lang="en">
          <head>
            <meta charset="UTF-8" />
            <title>Tabby</title>
          </head>
          <body>
            <h2>The chat panel is not available:</h2>
            <ul>
              <li>Update to <a href="https://github.com/TabbyML/tabby/releases" target="_blank"> the latest version of Tabby server.</a> </li>
              <li>You have to launch the server with the chat model enabled, for example, <code>--chat-model CodeQwen-7B-Chat</code>. </li>
              <li>The webserver feature needs to be enabled; do not use <code>--no-webserver</code>. </li>
            </ul>
          </body>
        </html>
      `;
    }
    const endpoint = serverInfo.config.endpoint;
    const styleUri = this.webview?.webview.asWebviewUri(
      Uri.joinPath(this.context.extensionUri, "assets", "chat-panel.css"),
    );
    return `
      <!DOCTYPE html>
      <html lang="en">
        <!--hash: ${hashObject(serverInfo)}-->
        <head>
          <meta charset="UTF-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <title>Tabby</title>
          <link href="${endpoint}" rel="preconnect">
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
                chatIframe.contentWindow.postMessage({ style: parentHtmlStyle }, "${endpoint}");
            
                let themeClass = document.body.className === 'vscode-dark' ? 'dark' : 'light'
                themeClass += ' vscode'
                chatIframe.contentWindow.postMessage({ themeClass: themeClass }, "${endpoint}");
              }
        
              window.addEventListener("message", (event) => {
                if (event.data) {
                  if (event.data.action === 'sync-theme') {
                    syncTheme();
                    return;
                  }

                  if (event.data.data) {
                    chatIframe.contentWindow.postMessage(event.data.data[0], "${endpoint}");
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
            src="${endpoint}/chat?from=vscode"
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
