import {
  ExtensionContext,
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
  Webview,
  ColorThemeKind,
} from "vscode";
import type { ServerApi, ChatMessage, Context, NavigateOpts, OnLoadedParams } from "tabby-chat-panel";
import { TABBY_CHAT_PANEL_API_VERSION } from "tabby-chat-panel";
import hashObject from "object-hash";
import * as semver from "semver";
import type { ServerInfo } from "tabby-agent";
import type { AgentFeature as Agent } from "../lsp/AgentFeature";
import { GitProvider } from "../git/GitProvider";
import { createClient } from "./chatPanel";
import { isBrowser } from "../env";

export class WebviewHelper {
  webview?: Webview;
  client?: ServerApi;
  private pendingMessages: ChatMessage[] = [];
  private pendingRelevantContexts: Context[] = [];
  private isChatPageDisplayed = false;

  constructor(
    private readonly context: ExtensionContext,
    private readonly agent: Agent,
    private readonly logger: LogOutputChannel,
    private readonly gitProvider: GitProvider,
  ) {}

  static getColorThemeString(kind: ColorThemeKind) {
    switch (kind) {
      case ColorThemeKind.Light:
      case ColorThemeKind.HighContrastLight:
        return "light";
      case ColorThemeKind.Dark:
      case ColorThemeKind.HighContrast:
        return "dark";
    }
  }

  static async resolveDocument(
    logger: LogOutputChannel,
    folders: readonly WorkspaceFolder[] | undefined,
    filepath: string,
  ): Promise<TextDocument | null> {
    if (filepath.startsWith("file://")) {
      const absoluteFilepath = Uri.parse(filepath, true);
      return workspace.openTextDocument(absoluteFilepath);
    }

    if (!folders) {
      return null;
    }

    for (const root of folders) {
      const absoluteFilepath = Uri.joinPath(root.uri, filepath);
      try {
        return await workspace.openTextDocument(absoluteFilepath);
      } catch (err) {
        // Do nothing, file doesn't exists.
      }
    }

    logger.info("File not found in workspace folders, trying with findFiles...");

    const files = await workspace.findFiles(filepath, undefined, 1);
    if (files[0]) {
      return workspace.openTextDocument(files[0]);
    }

    return null;
  }

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

    const { filepath, git_url } = WebviewHelper.resolveFilePathAndGitUrl(uri, gitProvider);

    return {
      kind: "file",
      content: alignIndent(text),
      range: {
        start: editor.selection.start.line + 1,
        end: editor.selection.end.line + 1,
      },
      filepath,
      git_url,
    };
  }

  static resolveFilePathAndGitUrl(uri: Uri, gitProvider: GitProvider): { filepath: string; git_url: string } {
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
      filepath: filePath.startsWith("/") ? filePath.substring(1) : filePath,
      git_url: remoteUrl ?? "",
    };
  }

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

  public checkChatPanelApiVersion(version: string): string | undefined {
    const serverApiVersion = semver.coerce(version);
    if (serverApiVersion) {
      this.logger.info(
        `Chat panel server API version: ${serverApiVersion}, client API version: ${TABBY_CHAT_PANEL_API_VERSION}`,
      );
      const clientApiMajorVersion = semver.major(TABBY_CHAT_PANEL_API_VERSION);
      const clientApiMinorVersion = semver.minor(TABBY_CHAT_PANEL_API_VERSION);
      const clientCompatibleRange = `~${clientApiMajorVersion}.${clientApiMinorVersion}`;
      if (semver.ltr(serverApiVersion, clientCompatibleRange)) {
        return "Please update your Tabby server to the latest version to use chat panel.";
      }
      if (semver.gtr(serverApiVersion, clientCompatibleRange)) {
        return "Please update the Tabby VSCode extension to the latest version to use chat panel.";
      }
    }
    return undefined;
  }

  public async displayChatPage(endpoint: string, opts?: { force: boolean }) {
    if (!endpoint) return;
    if (this.isChatPageDisplayed && !opts?.force) return;

    if (this.webview) {
      this.isChatPageDisplayed = true;
      const styleUri = this.webview.asWebviewUri(Uri.joinPath(this.context.extensionUri, "assets", "chat-panel.css"));

      const logoUri = this.webview.asWebviewUri(Uri.joinPath(this.context.extensionUri, "assets", "tabby.png"));

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
                  if (event.data) {
                    if (event.data.action === 'postMessageToChatPanel') {
                      chatIframe.contentWindow.postMessage(event.data.message, "*");
                    } else if (event.data.action === 'dispatchKeyboardEvent') {
                      window.dispatchEvent(new KeyboardEvent(event.data.type, event.data.event));
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

  public displayDisconnectedPage() {
    if (this.webview) {
      this.isChatPageDisplayed = false;

      const logoUri = this.webview.asWebviewUri(Uri.joinPath(this.context.extensionUri, "assets", "tabby.png"));
      const styleUri = this.webview.asWebviewUri(Uri.joinPath(this.context.extensionUri, "assets", "chat-panel.css"));
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

      const isMac = isBrowser
        ? navigator.userAgent.toLowerCase().includes("mac")
        : process.platform.toLowerCase().includes("darwin");
      this.client?.init({
        fetcherOptions: {
          authorization: serverInfo.config.token,
        },
        useMacOSKeyboardEventHandler: isMac,
      });
    }
  }

  public formatLineHashForCodeBrowser(
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

  public sendMessage(message: ChatMessage) {
    if (!this.client) {
      this.pendingMessages.push(message);
    } else {
      this.sendMessageToChatPanel(message);
    }
  }

  public addAgentEventListeners() {
    this.agent.on("didChangeStatus", async (status) => {
      if (status !== "disconnected") {
        const serverInfo = await this.agent.fetchServerInfo();
        this.displayChatPage(serverInfo.config.endpoint);
        this.refreshChatPage();
      } else if (this.isChatPageDisplayed) {
        this.displayDisconnectedPage();
      }
    });

    this.agent.on("didUpdateServerInfo", async () => {
      const serverInfo = await this.agent.fetchServerInfo();
      this.displayChatPage(serverInfo.config.endpoint, { force: true });
      this.refreshChatPage();
    });
  }

  public async displayPageBasedOnServerStatus() {
    // At this point, if the server instance is not set up, agent.status is 'notInitialized'.
    // We check for the presence of the server instance by verifying serverInfo.health["webserver"].
    const serverInfo = await this.agent.fetchServerInfo();
    if (serverInfo.health && serverInfo.health["webserver"]) {
      const serverInfo = await this.agent.fetchServerInfo();
      this.displayChatPage(serverInfo.config.endpoint);
    } else {
      this.displayDisconnectedPage();
    }
  }

  public createChatClient(webview: Webview) {
    return createClient(webview, {
      navigate: async (context: Context, opts?: NavigateOpts) => {
        if (opts?.openInEditor) {
          const document = await WebviewHelper.resolveDocument(
            this.logger,
            workspace.workspaceFolders,
            context.filepath,
          );
          if (!document) {
            throw new Error(`File not found: ${context.filepath}`);
          }

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
          const fileContext = WebviewHelper.getFileContextFromSelection({ editor, gitProvider: this.gitProvider });
          if (fileContext)
            // active selection
            chatMessage.activeContext = fileContext;
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
      onLoaded: (params: OnLoadedParams | undefined) => {
        if (params?.apiVersion) {
          const error = this.checkChatPanelApiVersion(params.apiVersion);
          if (error) {
            this.client?.showError({ content: error });
            return;
          }
        }
        setTimeout(() => {
          this.refreshChatPage();
        }, 300);
      },
      onCopy: (content) => {
        env.clipboard.writeText(content);
      },
      onKeyboardEvent: (type: string, event: KeyboardEventInit) => {
        this.logger.debug(`Dispatching keyboard event: ${type} ${JSON.stringify(event)}`);
        this.webview?.postMessage({ action: "dispatchKeyboardEvent", type, event });
      },
    });
  }
}

export function resolveFilePathAndGitUrl(uri: Uri, gitProvider: GitProvider): { filepath: string; git_url: string } {
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
    filepath: filePath.startsWith("/") ? filePath.substring(1) : filePath,
    git_url: remoteUrl ?? "",
  };
}
