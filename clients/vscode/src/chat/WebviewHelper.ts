
import {
  ExtensionContext,
  WebviewViewProvider,
  WebviewView,
  workspace,
  Uri,
  env,
  LogOutputChannel,
  TextEditor,
  window,
  Position,
  Range,
  Selection,
  TextEditorRevealType,
  ViewColumn,
  WorkspaceFolder,
  TextDocument,
  commands,
  ColorThemeKind,
  WebviewPanel,
  Webview
} from "vscode";
import type { ServerApi, ChatMessage, Context, NavigateOpts, FocusKeybinding } from "tabby-chat-panel";
import hashObject from "object-hash";
import * as semver from "semver";
import type { ServerInfo } from "tabby-agent";
import type { AgentFeature as Agent } from "../lsp/AgentFeature";
import { GitProvider } from "../git/GitProvider";
import { getLogger } from "../logger";
import { contributes } from "../../package.json";
import { parseKeybinding, readUserKeybindingsConfig } from "../util/KeybindingParser";

export class WebviewHelper {
  webview?: Webview;
  client?: ServerApi;
  private pendingMessages: ChatMessage[] = [];
  private pendingRelevantContexts: Context[] = [];
  private isChatPageDisplayed = false;
  // FIXME: this check is not compatible with the environment of a browser in macOS
  private isMac: boolean = env.appHost === "desktop" && process.platform === "darwin";
  
  constructor(
    private readonly context: ExtensionContext,
    private readonly agent: Agent,
    private readonly logger: LogOutputChannel,
    private readonly gitProvider: GitProvider,
  ) {}

  public setWebview(webview: Webview) {
    this.webview = webview;
  }

  public setClient(client: ServerApi) {
    this.client = client;
  }

  // Check if server is healthy and has the chat model enabled.
  //
  // Returns undefined if it's working, otherwise returns a message to display.
  public checkChatPanel(serverInfo: ServerInfo): string | undefined {
    if (!serverInfo.health) {
      return "Your Tabby server is not responding. Please check your server status.";
    }

    if (!serverInfo.health["webserver"] || !serverInfo.health["chat_model"]) {
      return "You need to launch the server with the chat model enabled; for example, use `--chat-model Qwen2-1.5B-Instruct`.";
    }

    const MIN_VERSION = "0.18.0";

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
      if (version && semver.lt(version, MIN_VERSION)) {
        return `Tabby Chat requires Tabby server version ${MIN_VERSION} or later. Your server is running version ${version}.`;
      }
    }

    return;
  }

  public async displayChatPage(endpoint: string, opts?: { force: boolean }) {
    if (!endpoint) return;
    if (this.isChatPageDisplayed && !opts?.force) return;

    if (this.webview) {
      this.isChatPageDisplayed = true;
      const styleUri = this.webview.asWebviewUri(
        Uri.joinPath(this.context.extensionUri, "assets", "chat-panel.css"),
      );

      const logoUri = this.webview.asWebviewUri(
        Uri.joinPath(this.context.extensionUri, "assets", "tabby.png"),
      );

      this.webview.html = `
        <!DOCTYPE html>
        <html lang="en">
          <!--hash: ${hashObject({ renderDate: new Date().toString() })}-->
          <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <link href="${endpoint}" rel="preconnect">
            <link href="${styleUri}" rel="stylesheet">
        
            <script defer>
              const vscode = acquireVsCodeApi();

              function getCssVariableValue(variableName) {
                const root = document.documentElement;
                return getComputedStyle(root).getPropertyValue(variableName).trim();
              }

              const syncTheme = () => {
                const parentHtmlStyle = document.documentElement.getAttribute('style');
                vscode.postMessage({
                  action: "sync-theme",
                  style: parentHtmlStyle
                })
              }

              const observer = new MutationObserver(function(mutations) {
                syncTheme();
              });
              
              observer.observe(document.documentElement, { attributes : true, attributeFilter : ['style'] });

              window.onload = function () {
                const chatIframe = document.getElementById("chat");
                const loadingOverlay = document.getElementById("loading-overlay");

                if (chatIframe) {
                  const fontSize = getCssVariableValue('--vscode-font-size');
                  const foreground = getCssVariableValue('--vscode-editor-foreground');

                  chatIframe.addEventListener('load', function() {
                    setTimeout(() => {
                      syncTheme()

                      setTimeout(() => {
                        loadingOverlay.style.display = 'none';
                        chatIframe.style.display = 'block';
                      }, 0)
                    }, 300)
                  });

                  chatIframe.src=encodeURI("${endpoint}/chat?client=vscode")
                }

                window.onfocus = (e) => {
                  if (chatIframe) {
                    // Directly call the focus method on the iframe's content window won't work in a focus event callback.
                    // Here we use a timeout to defer the focus call.
                    setTimeout(() => {
                      chatIframe.contentWindow.focus();
                    }, 0)
                  }
                }
                
                window.addEventListener("message", (event) => {
                  if (!chatIframe) return
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
            <main class='static-content' id='loading-overlay'>
              <div class='avatar'>
                <img src="${logoUri}" />
                <p>Tabby</p>
              </div>
              <p>Just a moment while we get things ready...</p>
              <span class='loader'></span>
            </main>
            <iframe
              id="chat"
              allow="clipboard-read; clipboard-write" />
          </body>
        </html>
      `;
    }
  }

  public async getFocusKeybinding(): Promise<FocusKeybinding | undefined> {
    const focusCommand = "tabby.chatView.focus";
    const defaultFocusKey = contributes.keybindings.find((cmd) => cmd.command === focusCommand);
    const defaultKeybinding = defaultFocusKey
      ? parseKeybinding(this.isMac && defaultFocusKey.mac ? defaultFocusKey.mac : defaultFocusKey.key)
      : undefined;

    const allKeybindings = await readUserKeybindingsConfig();
    const userShortcut = allKeybindings?.find((keybinding) => keybinding.command === focusCommand);

    return userShortcut ? parseKeybinding(userShortcut.key) : defaultKeybinding;
  }

  public displayDisconnectedPage() {
    if (this.webview) {
      this.isChatPageDisplayed = false;

      const logoUri = this.webview.asWebviewUri(
        Uri.joinPath(this.context.extensionUri, "assets", "tabby.png"),
      );
      const styleUri = this.webview.asWebviewUri(
        Uri.joinPath(this.context.extensionUri, "assets", "chat-panel.css"),
      );
      this.webview.html = `
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
              <h4 class='title'>Welcome to Tabby Chat!</h4>
              <p>To start chatting, please set up your Tabby server. Ensure that your Tabby server is properly configured and connected.</p>
            </main>
          </body>
        </html>
      `;
    }
  }

  public sendMessageToChatPanel(message: ChatMessage) {
    this.logger.info(`Sending message to chat panel: ${JSON.stringify(message)}`);
    this.client?.sendMessage(message);
  }

  public addRelevantContext(context: Context) {
    if (!this.client) {
      this.pendingRelevantContexts.push(context);
    } else {
      this.client?.addRelevantContext(context);
    }
  }

  public async refreshChatPage() {
    const agentStatus = this.agent.status;
    const serverInfo = await this.agent.fetchServerInfo();

    if (agentStatus === "unauthorized") {
      return this.client?.showError({
        content:
          "Before you can start chatting, please take a moment to set up your credentials to connect to the Tabby server.",
      });
    }

    const error = this.checkChatPanel(serverInfo);
    if (error) {
      this.client?.showError({ content: error });
      return;
    }

    this.pendingRelevantContexts.forEach((ctx) => this.addRelevantContext(ctx));
    this.pendingMessages.forEach((message) => this.sendMessageToChatPanel(message));

    if (serverInfo.config.token) {
      this.client?.cleanError();

      const focusKeybinding = await this.getFocusKeybinding();
      getLogger().info("focus key binding: ", focusKeybinding);

      this.client?.init({
        fetcherOptions: {
          authorization: serverInfo.config.token,
        },
        focusKey: focusKeybinding,
      });
    }
  }
}