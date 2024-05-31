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
  private isReady = false;

  constructor(
    private readonly context: ExtensionContext,
    private readonly agent: Agent,
  ) {}

  public async resolveWebviewView(webviewView: WebviewView) {
    this.webview = webviewView;
    this.isReady = this.agent.status === "ready";
    const extensionUri = this.context.extensionUri;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [extensionUri],
    };

    if (this.isReady) {
      await this.renderChatPage();
    } else {
      webviewView.webview.html = this.getWelcomeContent();
    }

    this.client = createClient(webviewView, {
      navigate: async (context: Context) => {
        if (context?.filepath && context?.git_url) {
          const serverInfo = await this.agent.fetchServerInfo();

          const url = new URL(`${serverInfo.config.endpoint}/files`);
          const searchParams = new URLSearchParams();
          searchParams.append("redirect_filepath", context.filepath);
          searchParams.append("redirect_git_url", context.git_url);
          searchParams.append("line", String(context.range.start));
          url.search = searchParams.toString();

          await env.openExternal(Uri.parse(url.toString()));
        }
      },
    });

    this.agent.on("didChangeStatus", async (status) => {
      if (status === "ready" && !this.isReady) {
        this.isReady = true;
        await this.renderChatPage();
      }
    });

    this.agent.on("didUpdateServerInfo", async (serverInfo: ServerInfo) => {
      await this.renderChatPage(serverInfo);
    });

    // The event will not be triggered during the initial rendering.
    webviewView.onDidChangeVisibility(async () => {
      if (webviewView.visible) {
        await this.initChatPage();
      }
    });

    webviewView.webview.onDidReceiveMessage(async (message) => {
      switch (message.action) {
        case "rendered": {
          await this.initChatPage();
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

  private async renderChatPage(serverInfo?: ServerInfo) {
    if (!serverInfo) {
      serverInfo = await this.agent.fetchServerInfo();
    }
    if (this.webview) {
      this.webview.webview.html = this.getWebviewContent(serverInfo);
    }
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

  private async initChatPage() {
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

  private getWebviewContent(serverInfo: ServerInfo) {
    if (!this.isChatPanelAvailable(serverInfo)) {
      return this.getStaticContent(`
        <h4 class='title'>Tabby is not available</h4>
        <p>Please update to <a href="https://github.com/TabbyML/tabby/releases" target="_blank">the latest version</a> of the Tabby server.</p>
        <p>You also need to launch the server with the chat model enabled; for example, use <code>--chat-model Mistral-7B</code>.</p>
      `);
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
                  if (event.data.action === 'copy') {
                    vscode.postMessage(event.data);
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
            allow="clipboard-read; clipboard-write"
            onload="iframeLoaded(this)" />
        </body>
      </html>
    `;
  }

  // The content is displayed before the server is ready
  private getWelcomeContent() {
    return this.getStaticContent(`
      <h4 class='title'>Welcome to Tabby Chat!</h4>
      <p>Before you can start chatting, please take a moment to set up your credentials to connect to the Tabby server.</p>
    `);
  }

  private getStaticContent(htmlContent: string) {
    const logoUri = this.webview?.webview.asWebviewUri(Uri.joinPath(this.context.extensionUri, "assets", "tabby.png"));
    const styleUri = this.webview?.webview.asWebviewUri(
      Uri.joinPath(this.context.extensionUri, "assets", "chat-panel.css"),
    );
    return `
      <!DOCTYPE html>
      <html lang="en">
        <head>
          <meta charset="UTF-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <link href="${styleUri}" rel="stylesheet">
        </head>
        <body>
          <main class='static-content'>
            <div class='avatar'>
              <img src="${logoUri}" />
              <p>Tabby</p>
            </div>
            ${htmlContent}
          </main>
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
