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
} from "vscode";
import type { ServerApi, ChatMessage, Context, NavigateOpts } from "tabby-chat-panel";
import hashObject from "object-hash";
import * as semver from "semver";
import type { ServerInfo } from "tabby-agent";
import type { AgentFeature as Agent } from "../lsp/AgentFeature";
import { createClient } from "./chatPanel";
import { GitProvider } from "../git/GitProvider";

export class ChatViewProvider implements WebviewViewProvider {
  webview?: WebviewView;
  client?: ServerApi;
  private pendingMessages: ChatMessage[] = [];
  private isChatPageDisplayed = false;

  constructor(
    private readonly context: ExtensionContext,
    private readonly agent: Agent,
    private readonly logger: LogOutputChannel,
    private readonly gitProvider: GitProvider,
  ) {}

  static getFileContextFromSelection({
    editor,
    gitProvider,
  }: {
    editor: TextEditor;
    gitProvider: GitProvider;
  }): Context | null {
    const alignIndent = (text: string) => {
      const lines = text.split("\n");
      const subsequentLines = lines.slice(1);

      // Determine the minimum indent for subsequent lines
      const minIndent = subsequentLines.reduce((min, line) => {
        const match = line.match(/^(\s*)/);
        const indent = match ? match[0].length : 0;
        return line.trim() ? Math.min(min, indent) : min;
      }, Infinity);

      // Remove the minimum indent
      const adjustedLines = lines.slice(1).map((line) => line.slice(minIndent));

      return [lines[0]?.trim(), ...adjustedLines].join("\n");
    };

    const uri = editor.document.uri;
    const text = editor.document.getText(editor.selection);
    if (!text) return null;

    const workspaceFolder = workspace.getWorkspaceFolder(uri);
    const repo = gitProvider.getRepository(uri);
    const remoteUrl = repo ? gitProvider.getDefaultRemoteUrl(repo) : undefined;
    let filePath = uri.toString(true);
    if (repo) {
      filePath = filePath.replace(repo.rootUri.toString(true), "");
    } else if (workspaceFolder) {
      filePath = filePath.replace(workspaceFolder.uri.toString(true), "");
    }

    return {
      kind: "file",
      content: alignIndent(text),
      range: {
        start: editor.selection.start.line + 1,
        end: editor.selection.end.line + 1,
      },
      filepath: filePath.startsWith("/") ? filePath.substring(1) : filePath,
      git_url: remoteUrl ?? "",
    };
  }

  // The method is called when the chat panel first opened
  public async resolveWebviewView(webviewView: WebviewView) {
    this.webview = webviewView;
    const extensionUri = this.context.extensionUri;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [extensionUri],
    };

    this.client = createClient(webviewView, {
      navigate: async (context: Context, opts?: NavigateOpts) => {
        if (opts?.openInEditor) {
          const files = await workspace.findFiles(context.filepath, null, 1);
          if (files[0]) {
            const document = await workspace.openTextDocument(files[0].path);
            const newEditor = await window.showTextDocument(document, {
              viewColumn: ViewColumn.Active,
              preview: false,
              preserveFocus: true,
            });

            // Move the cursor to the specified line
            const start = new Position(Math.max(0, context.range.start - 1), 0);
            const end = new Position(context.range.end, 0);
            newEditor.selection = new Selection(start, end);
            newEditor.revealRange(new Range(start, end), TextEditorRevealType.InCenter);
          }

          return;
        }

        if (context?.filepath && context?.git_url) {
          const serverInfo = await this.agent.fetchServerInfo();

          const url = new URL(`${serverInfo.config.endpoint}/files`);
          const searchParams = new URLSearchParams();

          searchParams.append("redirect_filepath", context.filepath);
          searchParams.append("redirect_git_url", context.git_url);
          url.search = searchParams.toString();

          const lineHash = this.formatLineHashForCodeBrowser(context.range);
          if (lineHash) {
            url.hash = lineHash;
          }

          await env.openExternal(Uri.parse(url.toString()));
        }
      },
      refresh: async () => {
        const serverInfo = await this.agent.fetchServerInfo();
        await this.displayChatPage(serverInfo.config.endpoint, { force: true });
        return;
      },
      onSubmitMessage: async (msg: string, relevantContext?: Context[]) => {
        const editor = window.activeTextEditor;
        const chatMessage: ChatMessage = {
          message: msg,
          relevantContext: [],
        };
        if (editor) {
          const fileContext = ChatViewProvider.getFileContextFromSelection({ editor, gitProvider: this.gitProvider });
          if (fileContext) chatMessage.relevantContext?.push(fileContext);
        }
        if (relevantContext) {
          chatMessage.relevantContext = chatMessage.relevantContext?.concat(relevantContext);
        }

        // FIXME: maybe deduplicate on chatMessage.relevantContext
        this.sendMessage(chatMessage);
      },
      onApplyInEditor: (content: string) => {
        const editor = window.activeTextEditor;
        if (editor) {
          const document = editor.document;
          const selection = editor.selection;

          // Determine the indentation for the content
          // The calculation is based solely on the indentation of the first line
          const lineText = document.lineAt(selection.start.line).text;
          const match = lineText.match(/^(\s*)/);
          const indent = match ? match[0] : "";

          // Determine the indentation for the content's first line
          // Note:
          // If using spaces, selection.start.character = 1 means 1 space
          // If using tabs, selection.start.character = 1 means 1 tab
          const indentUnit = indent[0];
          const indentAmountForTheFirstLine = Math.max(indent.length - selection.start.character, 0);
          const indentForTheFirstLine = indentUnit?.repeat(indentAmountForTheFirstLine) || "";

          // Indent the content
          const indentedContent = indentForTheFirstLine + content.replaceAll("\n", "\n" + indent);

          // Apply into the editor
          editor.edit((editBuilder) => {
            editBuilder.replace(selection, indentedContent);
          });
        }
      },
    });

    // At this point, if the server instance is not set up, agent.status is 'notInitialized'.
    // We check for the presence of the server instance by verifying serverInfo.health["webserver"].
    const serverInfo = await this.agent.fetchServerInfo();
    if (serverInfo.health && serverInfo.health["webserver"]) {
      const serverInfo = await this.agent.fetchServerInfo();
      this.displayChatPage(serverInfo.config.endpoint);
    } else {
      this.displayDisconnectedPage();
    }

    this.agent.on("didChangeStatus", async (status) => {
      if (status !== "disconnected") {
        const serverInfo = await this.agent.fetchServerInfo();
        this.displayChatPage(serverInfo.config.endpoint);
        this.refreshChatPage();
      } else if (this.isChatPageDisplayed) {
        this.displayDisconnectedPage();
      }
    });

    this.agent.on("didUpdateServerInfo", () => {
      this.refreshChatPage();
    });

    // The event will not be triggered during the initial rendering.
    webviewView.onDidChangeVisibility(() => {
      if (webviewView.visible) {
        this.refreshChatPage();
      }
    });

    webviewView.webview.onDidReceiveMessage((message) => {
      switch (message.action) {
        case "rendered": {
          setTimeout(() => {
            this.refreshChatPage();
          }, 300);
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

  private async refreshChatPage() {
    const agentStatus = this.agent.status;
    const serverInfo = await this.agent.fetchServerInfo();

    if (agentStatus === "unauthorized") {
      return this.client?.showError({
        content:
          "Before you can start chatting, please take a moment to set up your credentials to connect to the Tabby server.",
      });
    }

    if (!this.isChatPanelAvailable(serverInfo)) {
      this.client?.showError({
        content:
          "Please update to the latest release of the Tabby server.\n\nYou also need to launch the server with the chat model enabled; for example, use `--chat-model Qwen2-1.5B-Instruct`.",
      });
      return;
    }

    this.pendingMessages.forEach((message) => this.sendMessageToChatPanel(message));
    if (serverInfo.config.token) {
      this.client?.cleanError();
      // Duplicate init won't break or reload the current chat page
      this.client?.init({
        fetcherOptions: {
          authorization: serverInfo.config.token,
        },
      });
    }
  }

  private async displayChatPage(endpoint: string, opts?: { force: boolean }) {
    if (!endpoint) return;
    if (this.isChatPageDisplayed && !opts?.force) return;

    if (this.webview) {
      this.isChatPageDisplayed = true;
      const styleUri = this.webview?.webview.asWebviewUri(
        Uri.joinPath(this.context.extensionUri, "assets", "chat-panel.css"),
      );

      this.webview.webview.html = `
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

              function getTheme () {
                return document.body.className === 'vscode-dark' ? 'dark' : 'light'
              }

              function getCssVariableValue(variableName) {
                const root = document.documentElement;
                return getComputedStyle(root).getPropertyValue(variableName).trim();
              }

              const syncTheme = () => {
                const chatIframe = document.getElementById("chat");
                if (!chatIframe) return

                const parentHtmlStyle = document.documentElement.getAttribute('style');
                chatIframe.contentWindow.postMessage({ style: parentHtmlStyle }, "${endpoint}");

                let themeClass = getTheme()
                themeClass += ' vscode'
                chatIframe.contentWindow.postMessage({ themeClass: themeClass }, "${endpoint}");
              }

              window.onload = function () {
                const chatIframe = document.getElementById("chat");

                if (chatIframe) {
                  const fontSize = getCssVariableValue('--vscode-font-size');
                  const foreground = getCssVariableValue('--vscode-editor-foreground');
                  const theme = getTheme()

                  const clientQuery = "&client=vscode"
                  const themeQuery = "&theme=" + theme
                  const fontSizeQuery = "&font-size=" + fontSize
                  const foregroundQuery = "&foreground=" + foreground.replace('#', '')
      
                  chatIframe.addEventListener('load', function() {
                    vscode.postMessage({ action: 'rendered' });
                    setTimeout(() => {
                      syncTheme()
                    }, 300)
                  });

                  chatIframe.src=encodeURI("${endpoint}/chat?" + clientQuery + themeQuery + fontSizeQuery + foregroundQuery)
                }
                
                window.addEventListener("message", (event) => {
                  if (!chatIframe) return
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
              allow="clipboard-read; clipboard-write" />
          </body>
        </html>
      `;
    }
  }

  private displayDisconnectedPage() {
    if (this.webview) {
      this.isChatPageDisplayed = false;

      const logoUri = this.webview?.webview.asWebviewUri(
        Uri.joinPath(this.context.extensionUri, "assets", "tabby.png"),
      );
      const styleUri = this.webview?.webview.asWebviewUri(
        Uri.joinPath(this.context.extensionUri, "assets", "chat-panel.css"),
      );
      this.webview.webview.html = `
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

  public getWebview() {
    return this.webview;
  }

  public sendMessage(message: ChatMessage) {
    if (!this.client) {
      this.pendingMessages.push(message);
    } else {
      this.sendMessageToChatPanel(message);
    }
  }

  public addClientSelectedContext(context: Context) {
    this.client?.addClientSelectedContext(context);
  }

  private sendMessageToChatPanel(message: ChatMessage) {
    this.logger.info(`Sending message to chat panel: ${JSON.stringify(message)}`);
    this.client?.sendMessage(message);
  }

  private formatLineHashForCodeBrowser(
    range:
      | {
          start: number;
          end?: number;
        }
      | undefined,
  ): string {
    if (!range) return "";
    const { start, end } = range;
    if (typeof start !== "number") return "";
    if (start === end) return `L${start}`;
    return [start, end]
      .map((num) => (typeof num === "number" ? `L${num}` : undefined))
      .filter((o) => o !== undefined)
      .join("-");
  }
}
