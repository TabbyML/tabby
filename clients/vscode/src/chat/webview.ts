import {
  commands,
  env,
  window,
  workspace,
  Disposable,
  ExtensionContext,
  Uri,
  TextEditor,
  Range,
  TextDocument,
  Webview,
  ColorThemeKind,
  ProgressLocation,
  Location,
  LocationLink,
  Position,
} from "vscode";
import { TABBY_CHAT_PANEL_API_VERSION } from "tabby-chat-panel";
import type {
  ServerApi,
  ChatCommand,
  EditorContext,
  OnLoadedParams,
  LookupSymbolHint,
  SymbolInfo,
  FileLocation,
  GitRepository,
  LookupDefinitionsHint,
} from "tabby-chat-panel";
import * as semver from "semver";
import type { StatusInfo, Config } from "tabby-agent";
import type { GitProvider } from "../git/GitProvider";
import type { Client as LspClient } from "../lsp/Client";
import { createClient } from "./createClient";
import { isBrowser } from "../env";
import { getLogger } from "../logger";
import { getFileContextFromSelection } from "./fileContext";
import {
  chatPanelFilepathToLocalUri,
  chatPanelLocationToVSCodeRange,
  isValidForSyncActiveEditorSelection,
  convertDefinitionToSymbolInfo,
  getDefinitionLocations,
} from "./utils";
import mainHtml from "./html/main.html";
import errorHtml from "./html/error.html";
import { filterSymbolInfosByContextAndOverlap } from "./definitions";

export class ChatWebview {
  private readonly logger = getLogger("ChatWebView");
  private disposables: Disposable[] = [];
  private webview: Webview | undefined = undefined;
  private client: ServerApi | undefined = undefined;

  // Once the chat iframe is loaded, the `onLoaded` should be called from server side later,
  // and we can start to initialize the chat panel.
  // So we set a timeout here to ensure the `onLoaded` is called, otherwise we will show an error.
  private onLoadedTimeout: NodeJS.Timeout | undefined = undefined;

  // Mark the `onLoaded` is called. Before this, we should schedule the actions like
  // `addRelevantContext` and `executeCommand` as pending actions.
  private chatPanelLoaded = false;

  // Pending actions to perform after the chat panel is initialized.
  private pendingActions: (() => Promise<void>)[] = [];

  // The current server config used to load the chat panel.
  private currentConfig: Config["server"] | undefined = undefined;

  // A number to ensure the html is reloaded when assigned a new value
  private reloadCount = 0;

  // A callback list for `isFocused` method
  private pendingFocusCheckCallbacks: ((focused: boolean) => void)[] = [];

  constructor(
    private readonly context: ExtensionContext,
    private readonly lspClient: LspClient,
    private readonly gitProvider: GitProvider,
  ) {}

  async init(webview: Webview) {
    webview.options = {
      enableScripts: true,
      enableCommandUris: true,
    };
    this.webview = webview;

    this.client = this.createChatPanelApiClient();

    const statusListener = () => {
      this.checkStatusAndLoadContent();
    };
    this.lspClient.status.on("didChange", statusListener);
    this.disposables.push(
      new Disposable(() => {
        this.lspClient.status.off("didChange", statusListener);
      }),
    );
    this.checkStatusAndLoadContent();

    this.disposables.push(
      window.onDidChangeActiveTextEditor((editor) => {
        if (this.chatPanelLoaded) {
          this.syncActiveEditorSelection(editor);
        }
      }),
    );
    this.disposables.push(
      window.onDidChangeTextEditorSelection((event) => {
        if (event.textEditor === window.activeTextEditor && this.chatPanelLoaded) {
          this.syncActiveEditorSelection(event.textEditor);
        }
      }),
    );

    this.disposables.push(
      webview.onDidReceiveMessage((event) => {
        switch (event.action) {
          case "chatIframeLoaded": {
            this.onLoadedTimeout = setTimeout(() => {
              const endpoint = this.currentConfig?.endpoint ?? "";
              if (!endpoint) {
                this.checkStatusAndLoadContent();
              } else {
                const command = `command:tabby.openExternal?${encodeURIComponent(`["${endpoint}/chat"]`)}`;
                this.loadErrorPage(
                  `Failed to load the chat panel. <br/>Please check your network to ensure access to <a href='${command}'>${endpoint}/chat</a>. <br/><br/><a href='command:tabby.reconnectToServer'><b>Reload</b></a>`,
                );
              }
            }, 10000);
            return;
          }
          case "syncStyle": {
            this.client?.updateTheme(event.style, this.getColorThemeString());
            return;
          }
          case "checkFocusedResult": {
            this.pendingFocusCheckCallbacks.forEach((cb) => cb(event.focused));
            this.pendingFocusCheckCallbacks = [];
            return;
          }
        }
      }),
    );
  }

  async dispose() {
    this.disposables.forEach((d) => d.dispose());
    this.disposables = [];
  }

  async isFocused(): Promise<boolean> {
    const webview = this.webview;
    if (!webview) {
      return false;
    }
    return new Promise((resolve) => {
      webview.postMessage({ action: "checkFocused" });
      this.pendingFocusCheckCallbacks.push(resolve);
    });
  }

  async addRelevantContext(context: EditorContext) {
    if (this.client && this.chatPanelLoaded) {
      this.logger.info(`Adding relevant context: ${context}`);
      this.client.addRelevantContext(context);
    } else {
      this.pendingActions.push(async () => {
        this.logger.info(`Adding pending relevant context: ${context}`);
        await this.client?.addRelevantContext(context);
      });
    }
  }

  async executeCommand(command: ChatCommand) {
    if (this.client && this.chatPanelLoaded) {
      this.logger.info(`Executing command: ${command}`);
      this.client.executeCommand(command);
    } else {
      this.pendingActions.push(async () => {
        this.logger.info(`Executing pending command: ${command}`);
        await this.client?.executeCommand(command);
      });
    }
  }

  private async getDefinitionLocations(uri: Uri, position: Position) {
    return await commands.executeCommand<Location[] | LocationLink[]>(
      "vscode.executeDefinitionProvider",
      uri,
      position,
    );
  }

  private createChatPanelApiClient(): ServerApi | undefined {
    const webview = this.webview;
    if (!webview) {
      return undefined;
    }
    return createClient(webview, {
      refresh: async () => {
        commands.executeCommand("tabby.reconnectToServer");
        return;
      },

      onApplyInEditor: async (content: string) => {
        const editor = window.activeTextEditor;
        if (!editor) {
          window.showErrorMessage("No active editor found.");
          return;
        }
        await this.applyInEditor(editor, content);
      },

      onApplyInEditorV2: async (content: string, opts?: { languageId: string; smart: boolean }) => {
        const editor = window.activeTextEditor;
        if (!editor) {
          window.showErrorMessage("No active editor found.");
          return;
        }
        if (!opts || !opts.smart) {
          await this.applyInEditor(editor, content);
        } else if (editor.document.languageId !== opts.languageId) {
          this.logger.debug("Editor's languageId:", editor.document.languageId, "opts.languageId:", opts.languageId);
          await this.applyInEditor(editor, content);
          window.showInformationMessage("The active editor is not in the correct language. Did normal apply.");
        } else {
          await this.smartApplyInEditor(editor, content);
        }
      },

      onLoaded: async (params: OnLoadedParams | undefined) => {
        if (this.onLoadedTimeout) {
          clearTimeout(this.onLoadedTimeout);
          this.onLoadedTimeout = undefined;
        }

        if (params?.apiVersion) {
          const error = this.checkChatPanelApiVersion(params.apiVersion);
          if (error) {
            this.loadErrorPage(error);
            return;
          }
        }

        this.chatPanelLoaded = true;

        // 1. Sync the active editor selection
        // 2. Send pending actions
        // 3. Call the client's init method
        // 4. Show the chat panel (call syncStyle underlay)
        await this.syncActiveEditorSelection(window.activeTextEditor);

        this.pendingActions.forEach(async (fn) => {
          await fn();
        });
        this.pendingActions = [];

        const isMac = isBrowser
          ? navigator.userAgent.toLowerCase().includes("mac")
          : process.platform.toLowerCase().includes("darwin");

        await this.client?.init({
          fetcherOptions: {
            authorization: this.currentConfig?.token ?? "",
          },
          useMacOSKeyboardEventHandler: isMac,
        });

        this.webview?.postMessage({ action: "showChatPanel" });
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
              // get definitions
              const locations = await this.getDefinitionLocations(document.uri, position);
              if (!locations || locations.length === 0 || !locations[0]) {
                continue;
              }

              return convertDefinitionToSymbolInfo(document, position, locations[0], this.gitProvider);
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

        const targetRange = chatPanelLocationToVSCodeRange(fileLocation.location) ?? new Range(0, 0, 0, 0);
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

        const workspaceFolder = workspace.workspaceFolders ?? [];
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

      lookupDefinitions: async (context: LookupDefinitionsHint): Promise<SymbolInfo[]> => {
        if (!context?.filepath) {
          this.logger.info("lookupDefinitions: Missing filepath in context.");
          return [];
        }

        // convert ChatPanel filepath to a local URI
        const uri = chatPanelFilepathToLocalUri(context.filepath, this.gitProvider);
        if (!uri) {
          this.logger.info("lookupDefinitions: Could not resolve local URI for:", context.filepath);
          return [];
        }

        // open the document
        let document;
        try {
          document = await workspace.openTextDocument(uri);
        } catch (e) {
          this.logger.info("lookupDefinitions: Can't open file:", uri);
          return [];
        }

        // determine the snippet range and get its text
        const snippetRange = this.getSnippetRange(document, context);
        const snippetText = document.getText(snippetRange);

        // split the text into words
        const words = snippetText.split(/\b/);

        // Use an offset accumulator to track each word's position
        let offset = 0;

        // Map each word to an async definition lookup
        const tasks = words.map((rawWord) => {
          const currentOffset = offset;
          offset += rawWord.length;

          const trimmedWord = rawWord.trim();
          if (!trimmedWord || trimmedWord.match(/^\W+$/)) {
            return Promise.resolve<SymbolInfo[]>([]);
          }

          const position = document.positionAt(document.offsetAt(snippetRange.start) + currentOffset);

          return getDefinitionLocations(document.uri, position)
            .then((definitions) => {
              if (!definitions || definitions.length === 0) {
                return [];
              }
              const result: SymbolInfo[] = [];
              definitions.forEach((def) => {
                const info = convertDefinitionToSymbolInfo(document, position, def, this.gitProvider);
                if (info) {
                  result.push(info);
                }
              });
              return result;
            })
            .catch((err) => {
              this.logger.error(`lookupDefinitions: DefinitionProvider error: ${err}`);
              return [];
            });
        });

        // await all lookups in parallel and flatten the results
        const symbolInfosArrays = await Promise.all(tasks);
        const symbolInfos = symbolInfosArrays.flat();

        // filter and merge final results
        return filterSymbolInfosByContextAndOverlap(symbolInfos, context);
      },
    });
  }
  /**
   * Helper: decide snippet range from context.location or entire doc.
   */
  private getSnippetRange(document: TextDocument, context: LookupDefinitionsHint): Range {
    if (!context.location) {
      return new Range(0, 0, document.lineCount, 0);
    }
    const vsRange = chatPanelLocationToVSCodeRange(context.location);
    if (!vsRange || vsRange.isEmpty) {
      return new Range(0, 0, document.lineCount, 0);
    }
    return vsRange;
  }

  private checkStatusAndLoadContent() {
    const statusInfo = this.lspClient.status.current;
    const error = this.checkStatusInfo(statusInfo);
    if (error) {
      this.currentConfig = undefined;
      this.loadErrorPage(error);
      return;
    }
    const config = this.lspClient.agentConfig.current;
    if (!config) {
      this.currentConfig = undefined;
      this.loadErrorPage("Cannot get the server configuration.");
      return;
    }
    if (this.currentConfig?.endpoint !== config.server.endpoint || this.currentConfig?.token !== config.server.token) {
      this.currentConfig = config.server;
      this.loadChatPanel();
    }
  }

  private getUriStylesheet() {
    return (
      this.webview?.asWebviewUri(Uri.joinPath(this.context.extensionUri, "assets", "chat-panel.css")).toString() ?? ""
    );
  }

  private getUriAvatarTabby() {
    return this.webview?.asWebviewUri(Uri.joinPath(this.context.extensionUri, "assets", "tabby.png")).toString() ?? "";
  }

  private loadChatPanel() {
    const webview = this.webview;
    if (!webview) {
      return;
    }
    this.chatPanelLoaded = false;
    this.reloadCount += 1;
    webview.html = mainHtml
      .replace(/{{RELOAD_COUNT}}/g, this.reloadCount.toString())
      .replace(/{{SERVER_ENDPOINT}}/g, this.currentConfig?.endpoint ?? "")
      .replace(/{{URI_STYLESHEET}}/g, this.getUriStylesheet())
      .replace(/{{URI_AVATAR_TABBY}}/g, this.getUriAvatarTabby());
  }

  private loadErrorPage(message: string) {
    const webview = this.webview;
    if (!webview) {
      return;
    }
    this.chatPanelLoaded = false;
    this.reloadCount += 1;
    webview.html = errorHtml
      .replace(/{{RELOAD_COUNT}}/g, this.reloadCount.toString())
      .replace(/{{URI_STYLESHEET}}/g, this.getUriStylesheet())
      .replace(/{{URI_AVATAR_TABBY}}/g, this.getUriAvatarTabby())
      .replace(/{{ERROR_MESSAGE}}/g, message);
  }

  // Returns undefined if no error, otherwise returns the error message
  private checkStatusInfo(statusInfo: StatusInfo | undefined): string | undefined {
    if (!statusInfo || statusInfo.status === "connecting") {
      return 'Connecting to the Tabby server...<br/><span class="loader"></span>';
    }

    if (statusInfo.status === "unauthorized") {
      return "Your token is invalid.<br/><a href='command:tabby.updateToken'><b>Update Token</b></a>";
    }

    if (statusInfo.status === "disconnected") {
      return "Failed to connect to the Tabby server.<br/><a href='command:tabby.connectToServer'><b>Connect To Server</b></a>";
    }

    const health = statusInfo.serverHealth;
    if (!health) {
      return "Cannot get the health status of the Tabby server.";
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

    return undefined;
  }

  private checkChatPanelApiVersion(version: string): string | undefined {
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

  private async applyInEditor(editor: TextEditor, content: string) {
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
    const indentForTheFirstLine = indentUnit?.repeat(indentAmountForTheFirstLine) ?? "";
    // Indent the content
    const indentedContent = indentForTheFirstLine + content.replaceAll("\n", "\n" + indent);

    // Apply into the editor
    await editor.edit((editBuilder) => {
      editBuilder.replace(selection, indentedContent);
    });
  }

  private async smartApplyInEditor(editor: TextEditor, content: string) {
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
  }

  private async syncActiveEditorSelection(editor: TextEditor | undefined) {
    if (!editor || !isValidForSyncActiveEditorSelection(editor)) {
      await this.client?.updateActiveSelection(null);
      return;
    }

    const fileContext = await getFileContextFromSelection(editor, this.gitProvider);
    await this.client?.updateActiveSelection(fileContext);
  }

  private getColorThemeString() {
    switch (window.activeColorTheme.kind) {
      case ColorThemeKind.Light:
      case ColorThemeKind.HighContrastLight:
        return "light";
      case ColorThemeKind.Dark:
      case ColorThemeKind.HighContrast:
        return "dark";
    }
  }
}
