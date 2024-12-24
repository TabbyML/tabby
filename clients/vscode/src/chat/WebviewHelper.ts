import {
  ExtensionContext,
  Uri,
  env,
  TextEditor,
  window,
  Range,
  Selection,
  TextDocument,
  Webview,
  ColorThemeKind,
  ProgressLocation,
  commands,
  Location,
  LocationLink,
  workspace,
} from "vscode";
import type {
  ServerApi,
  ChatCommand,
  EditorContext,
  OnLoadedParams,
  LookupSymbolHint,
  SymbolInfo,
  FileLocation,
  GitRepository,
} from "tabby-chat-panel";
import { TABBY_CHAT_PANEL_API_VERSION } from "tabby-chat-panel";
import hashObject from "object-hash";
import * as semver from "semver";
import type { StatusInfo } from "tabby-agent";
import type { LogOutputChannel } from "../logger";
import { GitProvider } from "../git/GitProvider";
import { createClient } from "./chatPanel";
import { Client as LspClient } from "../lsp/Client";
import { isBrowser } from "../env";
import { getFileContextFromSelection } from "./fileContext";
import {
  localUriToChatPanelFilepath,
  chatPanelFilepathToLocalUri,
  vscodePositionToChatPanelPosition,
  vscodeRangeToChatPanelPositionRange,
  chatPanelLocationToVSCodeRange,
} from "./utils";

export class WebviewHelper {
  webview?: Webview;
  client?: ServerApi;
  private pendingActions: (() => Promise<void>)[] = [];
  private isChatPageDisplayed = false;

  constructor(
    private readonly context: ExtensionContext,
    private readonly lspClient: LspClient,
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

  public setWebview(webview: Webview) {
    this.webview = webview;
  }

  public setClient(client: ServerApi) {
    this.client = client;
  }

  // Check if server is healthy and has the chat model enabled.
  //
  // Returns undefined if it's working, otherwise returns a message to display.
  public checkChatPanel(statusInfo: StatusInfo | undefined): string | undefined {
    const health = statusInfo?.serverHealth;
    if (!health) {
      return "Your Tabby server is not responding. Please check your server status.";
    }

    if (!health["webserver"] || !health["chat_model"]) {
      return "You need to launch the server with the chat model enabled; for example, use `--chat-model Qwen2-1.5B-Instruct`.";
    }

    const MIN_VERSION = "0.18.0";

    if (health["version"]) {
      let version: semver.SemVer | undefined | null = undefined;
      if (typeof health["version"] === "string") {
        version = semver.coerce(health["version"]);
      } else if (
        typeof health["version"] === "object" &&
        "git_describe" in health["version"] &&
        typeof health["version"]["git_describe"] === "string"
      ) {
        version = semver.coerce(health["version"]["git_describe"]);
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

  public isSupportedSchemeForActiveSelection(scheme: string) {
    const supportedSchemes = ["file", "untitled"];
    return supportedSchemes.includes(scheme);
  }

  public async syncActiveSelectionToChatPanel(context: EditorContext | null) {
    try {
      await this.client?.updateActiveSelection(context);
    } catch {
      this.logger.log(
        {
          every: 100,
          level: "warn",
        },
        "Active selection sync failed. Please update your Tabby server to the latest version.",
      );
    }
  }

  public addRelevantContext(context: EditorContext) {
    if (this.client) {
      this.logger.info(`Adding relevant context: ${context}`);
      this.client.addRelevantContext(context);
    } else {
      this.pendingActions.push(async () => {
        this.logger.info(`Adding pending relevant context: ${context}`);
        await this.client?.addRelevantContext(context);
      });
    }
  }

  public executeCommand(command: ChatCommand) {
    if (this.client) {
      this.logger.info(`Executing command: ${command}`);
      this.client.executeCommand(command);
    } else {
      this.pendingActions.push(async () => {
        this.logger.info(`Executing pending command: ${command}`);
        await this.client?.executeCommand(command);
      });
    }
  }

  public async refreshChatPage() {
    const statusInfo = this.lspClient.status.current;

    if (statusInfo?.status === "unauthorized") {
      return this.client?.showError({
        content:
          "Before you can start chatting, please take a moment to set up your credentials to connect to the Tabby server.",
      });
    }

    const error = this.checkChatPanel(statusInfo);
    if (error) {
      this.client?.showError({ content: error });
      return;
    }

    await this.syncActiveSelection(window.activeTextEditor);

    this.pendingActions.forEach(async (fn) => {
      await fn();
    });
    this.pendingActions = [];

    const agentConfig = this.lspClient.agentConfig.current;
    if (agentConfig?.server.token) {
      this.client?.cleanError();

      const isMac = isBrowser
        ? navigator.userAgent.toLowerCase().includes("mac")
        : process.platform.toLowerCase().includes("darwin");
      await this.client?.init({
        fetcherOptions: {
          authorization: agentConfig.server.token,
        },
        useMacOSKeyboardEventHandler: isMac,
      });
    }
  }

  public async syncActiveSelection(editor: TextEditor | undefined) {
    if (!editor || !this.isSupportedSchemeForActiveSelection(editor.document.uri.scheme)) {
      await this.syncActiveSelectionToChatPanel(null);
      return;
    }

    const fileContext = await getFileContextFromSelection(editor, this.gitProvider);
    await this.syncActiveSelectionToChatPanel(fileContext);
  }

  public addAgentEventListeners() {
    this.lspClient.status.on("didChange", async (status: StatusInfo) => {
      const agentConfig = this.lspClient.agentConfig.current;
      if (agentConfig && status.serverHealth) {
        this.displayChatPage(agentConfig.server.endpoint);
        this.refreshChatPage();
      } else if (this.isChatPageDisplayed) {
        this.displayDisconnectedPage();
      }
    });
  }

  public addTextEditorEventListeners() {
    window.onDidChangeActiveTextEditor((e) => {
      this.syncActiveSelection(e);
    });

    window.onDidChangeTextEditorSelection((e) => {
      // This listener only handles text files.
      if (!this.isSupportedSchemeForActiveSelection(e.textEditor.document.uri.scheme)) {
        return;
      }
      this.syncActiveSelection(e.textEditor);
    });
  }

  public async displayPageBasedOnServerStatus() {
    const statusInfo = this.lspClient.status.current;
    const agentConfig = this.lspClient.agentConfig.current;
    if (statusInfo?.serverHealth && statusInfo?.serverHealth["webserver"] && agentConfig) {
      this.displayChatPage(agentConfig.server.endpoint, { force: true });
    } else {
      this.displayDisconnectedPage();
    }
  }

  public createChatClient(webview: Webview) {
    /*
      utility functions for createClient
    */
    const getIndentInfo = (document: TextDocument, selection: Selection) => {
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

      return { indent, indentForTheFirstLine };
    };

    const applyInEditor = (editor: TextEditor, content: string) => {
      const document = editor.document;
      const selection = editor.selection;
      const { indent, indentForTheFirstLine } = getIndentInfo(document, selection);
      // Indent the content
      const indentedContent = indentForTheFirstLine + content.replaceAll("\n", "\n" + indent);
      // Apply into the editor
      editor.edit((editBuilder) => {
        editBuilder.replace(selection, indentedContent);
      });
    };

    return createClient(webview, {
      refresh: async () => {
        const agentConfig = await this.lspClient.agentConfig.fetchAgentConfig();
        await this.displayChatPage(agentConfig.server.endpoint, { force: true });
        return;
      },
      onApplyInEditor: async (content: string) => {
        const editor = window.activeTextEditor;
        if (!editor) {
          window.showErrorMessage("No active editor found.");
          return;
        }
        applyInEditor(editor, content);
      },
      onApplyInEditorV2: async (content: string, opts?: { languageId: string; smart: boolean }) => {
        const smartApplyInEditor = async (editor: TextEditor, opts: { languageId: string; smart: boolean }) => {
          if (editor.document.languageId !== opts.languageId) {
            this.logger.debug("Editor's languageId:", editor.document.languageId, "opts.languageId:", opts.languageId);
            window.showInformationMessage("The active editor is not in the correct language. Did normal apply.");
            applyInEditor(editor, content);
            return;
          }

          this.logger.info("Smart apply in editor started.");
          this.logger.trace("Smart apply in editor with content:", { content });

          window.withProgress(
            {
              location: ProgressLocation.Notification,
              title: "Smart Apply in Progress",
              cancellable: true,
            },
            async (progress, token) => {
              progress.report({ increment: 0, message: "Applying smart edit..." });
              try {
                await this.lspClient.chat.provideSmartApplyEdit(
                  {
                    text: content,
                    location: {
                      uri: editor.document.uri.toString(),
                      range: {
                        start: { line: editor.selection.start.line, character: editor.selection.start.character },
                        end: { line: editor.selection.end.line, character: editor.selection.end.character },
                      },
                    },
                  },
                  token,
                );
              } catch (error) {
                if (error instanceof Error) {
                  window.showErrorMessage(error.message);
                } else {
                  window.showErrorMessage("An unknown error occurred");
                }
              }
            },
          );
        };

        const editor = window.activeTextEditor;
        if (!editor) {
          window.showErrorMessage("No active editor found.");
          return;
        }
        if (!opts || !opts.smart) {
          applyInEditor(editor, content);
        } else {
          smartApplyInEditor(editor, opts);
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
      lookupSymbol: async (symbol: string, hints?: LookupSymbolHint[] | undefined): Promise<SymbolInfo | undefined> => {
        if (!symbol.match(/^[a-zA-Z_][a-zA-Z0-9_]*$/)) {
          // Do not process invalid symbols
          return undefined;
        }
        /// FIXME: When no hints provided, try to use `vscode.executeWorkspaceSymbolProvider` to find the symbol.

        // Find the symbol in the hints
        for (const hint of hints ?? []) {
          if (!hint.filepath) {
            this.logger.debug("No filepath in the hint:", hint);
            continue;
          }
          const uri = chatPanelFilepathToLocalUri(hint.filepath, this.gitProvider);
          if (!uri) {
            continue;
          }

          let document: TextDocument;
          try {
            document = await workspace.openTextDocument(uri);
          } catch (error) {
            this.logger.debug("Failed to open document:", uri, error);
            continue;
          }
          if (!document) {
            continue;
          }

          const findSymbolInContent = async (
            content: string,
            offsetInDocument: number,
          ): Promise<SymbolInfo | undefined> => {
            // Add word boundary to perform exact match
            const matchRegExp = new RegExp(`\\b${symbol}\\b`, "g");
            let match;
            while ((match = matchRegExp.exec(content)) !== null) {
              const offset = offsetInDocument + match.index;
              const position = document.positionAt(offset);
              const locations = await commands.executeCommand<Location[] | LocationLink[]>(
                "vscode.executeDefinitionProvider",
                document.uri,
                position,
              );
              if (locations && locations.length > 0) {
                const location = locations[0];
                if (location) {
                  if ("targetUri" in location) {
                    const targetLocation = location.targetSelectionRange ?? location.targetRange;
                    return {
                      source: {
                        filepath: localUriToChatPanelFilepath(document.uri, this.gitProvider),
                        location: vscodePositionToChatPanelPosition(position),
                      },
                      target: {
                        filepath: localUriToChatPanelFilepath(location.targetUri, this.gitProvider),
                        location: vscodeRangeToChatPanelPositionRange(targetLocation),
                      },
                    };
                  } else if ("uri" in location) {
                    return {
                      source: {
                        filepath: localUriToChatPanelFilepath(document.uri, this.gitProvider),
                        location: vscodePositionToChatPanelPosition(position),
                      },
                      target: {
                        filepath: localUriToChatPanelFilepath(location.uri, this.gitProvider),
                        location: vscodeRangeToChatPanelPositionRange(location.range),
                      },
                    };
                  }
                }
              }
            }
            return undefined;
          };

          let symbolInfo: SymbolInfo | undefined;
          if (hint.location) {
            // Find in the hint location
            const location = chatPanelLocationToVSCodeRange(hint.location);
            if (location) {
              let range: Range;
              if (!location.isEmpty) {
                range = location;
              } else {
                // a empty range, create a new range with this line to the end of the file
                range = new Range(location.start.line, 0, document.lineCount, 0);
              }
              const content = document.getText(range);
              const offset = document.offsetAt(range.start);
              symbolInfo = await findSymbolInContent(content, offset);
            }
          }
          if (!symbolInfo) {
            // Fallback to find in full content
            const content = document.getText();
            symbolInfo = await findSymbolInContent(content, 0);
          }
          if (symbolInfo) {
            // Symbol found
            this.logger.debug(
              `Symbol found: ${symbol} with hints: ${JSON.stringify(hints)}: ${JSON.stringify(symbolInfo)}`,
            );
            return symbolInfo;
          }
        }
        this.logger.debug(`Symbol not found: ${symbol} with hints: ${JSON.stringify(hints)}`);
        return undefined;
      },
      openInEditor: async (fileLocation: FileLocation): Promise<boolean> => {
        const uri = chatPanelFilepathToLocalUri(fileLocation.filepath, this.gitProvider);
        if (!uri) {
          return false;
        }

        if (uri.scheme === "output") {
          try {
            await commands.executeCommand(`workbench.action.output.show.${uri.fsPath}`);
            return true;
          } catch (error) {
            this.logger.error("Failed to open output channel:", fileLocation, error);
            return false;
          }
        }

        const targetRange = fileLocation.location ? chatPanelLocationToVSCodeRange(fileLocation.location) ?? new Range(0, 0, 0, 0) : new Range(0, 0, 0, 0);
        try {
          await commands.executeCommand(
            "editor.action.goToLocations",
            uri,
            targetRange.start,
            [new Location(uri, targetRange)],
            "goto",
          );
          return true;
        } catch (error) {
          this.logger.error("Failed to go to location:", fileLocation, error);
          return false;
        }
      },
      openExternal: async (url: string) => {
        await env.openExternal(Uri.parse(url));
      },
      readWorkspaceGitRepositories: async (): Promise<GitRepository[]> => {
        const activeTextEditor = window.activeTextEditor;
        const infoList: GitRepository[] = [];
        let activeGitUrl: string | undefined;
        if (activeTextEditor) {
          const repo = this.gitProvider.getRepository(activeTextEditor.document.uri);
          if (repo) {
            const gitRemoteUrl = this.gitProvider.getDefaultRemoteUrl(repo);
            if (gitRemoteUrl) {
              infoList.push({
                url: gitRemoteUrl,
              });
            }
          }
        }

        const workspaceFolder = workspace.workspaceFolders || [];
        for (const folder of workspaceFolder) {
          const repo = this.gitProvider.getRepository(folder.uri);
          if (repo) {
            const gitRemoteUrl = this.gitProvider.getDefaultRemoteUrl(repo);
            if (gitRemoteUrl && gitRemoteUrl !== activeGitUrl) {
              infoList.push({
                url: gitRemoteUrl,
              });
            }
          }
        }
        return infoList;
      },
    });
  }
}
