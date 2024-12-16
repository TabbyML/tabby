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
  Position,
  Range,
} from "vscode";
import type {
  ServerApi,
  ChatMessage,
  Context,
  NavigateOpts,
  OnLoadedParams,
  LookupSymbolHint,
  SymbolInfo,
  FileLocation,
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
import { getFileContextFromSelection, showFileContext, openTextDocument } from "./fileContext";
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
  private pendingMessages: ChatMessage[] = [];
  private pendingRelevantContexts: Context[] = [];
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

  public sendMessageToChatPanel(message: ChatMessage) {
    this.logger.info(`Sending message to chat panel: ${JSON.stringify(message)}`);
    this.client?.sendMessage(message);
  }

  public async syncActiveSelectionToChatPanel(context: Context | null) {
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

  public addRelevantContext(context: Context) {
    if (!this.client) {
      this.pendingRelevantContexts.push(context);
    } else {
      this.client?.addRelevantContext(context);
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

    this.pendingRelevantContexts.forEach((ctx) => this.addRelevantContext(ctx));
    this.pendingMessages.forEach((message) => this.sendMessageToChatPanel(message));
    this.syncActiveSelection(window.activeTextEditor);

    const agentConfig = this.lspClient.agentConfig.current;
    if (agentConfig?.server.token) {
      this.client?.cleanError();

      const isMac = isBrowser
        ? navigator.userAgent.toLowerCase().includes("mac")
        : process.platform.toLowerCase().includes("darwin");
      this.client?.init({
        fetcherOptions: {
          authorization: agentConfig.server.token,
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

  public async syncActiveSelection(editor: TextEditor | undefined) {
    if (!editor || !this.isSupportedSchemeForActiveSelection(editor.document.uri.scheme)) {
      this.syncActiveSelectionToChatPanel(null);
      return;
    }

    const fileContext = await getFileContextFromSelection(editor, this.gitProvider);
    this.syncActiveSelectionToChatPanel(fileContext);
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

    // get definition locations
    const getDefinitionLocations = async (uri: Uri, position: Position) => {
      return await commands.executeCommand<LocationLink[]>("vscode.executeDefinitionProvider", uri, position);
    };

    return createClient(webview, {
      navigate: async (context: Context, opts?: NavigateOpts) => {
        if (opts?.openInEditor) {
          showFileContext(context, this.gitProvider);
          return;
        }

        if (context?.filepath && context?.git_url) {
          const agentConfig = this.lspClient.agentConfig.current;

          const url = new URL(`${agentConfig?.server.endpoint}/files`);
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
        const agentConfig = await this.lspClient.agentConfig.fetchAgentConfig();
        await this.displayChatPage(agentConfig.server.endpoint, { force: true });
        return;
      },
      onSubmitMessage: async (msg: string, relevantContext?: Context[]) => {
        const editor = window.activeTextEditor;
        const chatMessage: ChatMessage = {
          message: msg,
          relevantContext: [],
        };
        // FIXME: after synchronizing the activeSelection, perhaps there's no need to include activeSelection in the message.
        if (editor && this.isSupportedSchemeForActiveSelection(editor.document.uri.scheme)) {
          const fileContext = await getFileContextFromSelection(editor, this.gitProvider);
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
          const document = await openTextDocument({ filePath: uri.toString(true) }, this.gitProvider);
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
      onLookupDefinitions: async (context: Context): Promise<Context[]> => {
        if (!context?.filepath || !context?.range) {
          this.logger.info("Invalid context - missing required fields:", {
            filepath: !!context?.filepath,
            range: !!context?.range,
          });
          return [];
        }

        const workspaceRoot = context.filepath.split("/")[0];
        if (!workspaceRoot) {
          this.logger.info("Could not determine workspace root from filepath:", context.filepath);
          return [];
        }

        const document = await openTextDocument({ filePath: context.filepath }, this.gitProvider);
        if (!document) {
          this.logger.info(`File not found: ${context.filepath}`);
          return [];
        }

        const textRange = new Range(new Position(context.range.start, 0), new Position(context.range.end + 1, 0));

        const text = document.getText(textRange);
        if (!text) {
          this.logger.info("Empty text content for range:", textRange);
          return [];
        }

        let allResults: Context[] = [];
        const words = text.split(/\b/);

        for (let i = 0; i < words.length; i++) {
          const word = words[i]?.trim();
          if (!word || word.match(/^\W+$/)) {
            continue;
          }

          let pos = 0;
          let currentLine = context.range.start;
          let currentChar = 0;

          for (let j = 0; j < i; j++) {
            const word = words[j];
            if (word) {
              pos += word.length;
            }
          }

          const lines = text.slice(0, pos).split("\n");
          currentLine += lines.length - 1;
          if (lines.length > 1) {
            currentChar = lines[lines.length - 1]?.length ?? 0;
          } else {
            currentChar += pos;
          }

          try {
            const position = new Position(currentLine, currentChar);
            const locations = await getDefinitionLocations(document.uri, position);

            if (!locations?.[0]) {
              continue;
            }

            const location = locations[0];
            const targetUri = location.targetUri?.toString(true);
            if (!targetUri) {
              this.logger.info("Invalid targetUri for location:", location);
              continue;
            }

            const fullPath = targetUri.replace(/^file:\/\/\//, "");
            const relativePath = fullPath.split(`/${workspaceRoot}/`)[1];
            if (!relativePath) {
              this.logger.info("Could not extract relative path from:", { targetUri, workspaceRoot });
              continue;
            }

            const targetDocument = await openTextDocument(
              { filePath: `${workspaceRoot}/${relativePath}` },
              this.gitProvider,
            );
            if (!targetDocument) {
              this.logger.info(`Could not open target file: ${relativePath}`);
              continue;
            }

            if (!location.targetRange?.start || !location.targetRange?.end) {
              this.logger.info("Invalid target range in location:", location);
              continue;
            }

            const targetRange = new Range(
              new Position(location.targetRange.start.line, 0),
              new Position(location.targetRange.end.line + 1, 0),
            );

            const targetContent = targetDocument.getText(targetRange);
            if (!targetContent) {
              this.logger.info("Failed to get target content for range:", targetRange);
              continue;
            }

            allResults.push({
              kind: "file",
              filepath: `${workspaceRoot}/${relativePath}`,
              range: {
                start: location.targetRange.start.line,
                end: location.targetRange.end.line,
              },
              content: targetContent,
              git_url: context.git_url,
            });
          } catch (error) {
            this.logger.error(`Error looking up definition for word "${word}":`, error);
          }
        }

        // transform results
        // remove start 0 and end 0
        allResults = allResults.filter((result) => result.range?.start !== 0 && result.range?.end !== 0);

        // merge overlapping ranges
        const fileGroups = new Map<string, Context[]>();
        for (const result of allResults) {
          if (!result?.filepath) continue;
          const existing = fileGroups.get(result.filepath) || [];
          existing.push(result);
          fileGroups.set(result.filepath, existing);
        }

        const finalResults: Context[] = [];
        for (const [filepath, contexts] of fileGroups) {
          if (!contexts?.length) continue;

          if (contexts.length === 1) {
            const context = contexts[0];
            if (context) {
              finalResults.push(context);
            }
            continue;
          }

          contexts.sort((a, b) => (a.range?.start ?? 0) - (b.range?.start ?? 0));
          let current = contexts[0];

          // TODO(Sma1lboy): consider move range.ts to common packages
          // for handle all case including Range from vscode and vscode-languageserver, also our own define LineRange
          for (let i = 1; i < contexts.length; i++) {
            const next = contexts[i];
            if (!next?.range?.start || !next?.range?.end || !current?.range?.end || !current?.range?.start) {
              this.logger.info("Invalid range:", { current, next });
              continue;
            }

            if (next.range.start <= current.range.end + 1) {
              current = {
                ...current,
                range: {
                  start: Math.min(current.range.start, next.range.start),
                  end: Math.max(current.range.end, next.range.end),
                },
              };

              const targetDocument = await openTextDocument({ filePath: filepath }, this.gitProvider);
              if (targetDocument) {
                const mergedRange = new Range(
                  new Position(current.range.start, 0),
                  new Position(current.range.end + 1, 0),
                );
                const mergedContent = targetDocument.getText(mergedRange);
                if (mergedContent) {
                  current.content = mergedContent;
                }
              }
            } else {
              finalResults.push(current);
              current = next;
            }
          }

          if (current) {
            finalResults.push(current);
          }
        }

        return finalResults;
      },
      onLookupDefinitions: async (context: Context): Promise<Context[]> => {
        if (!context?.filepath || !context?.range) {
          this.logger.info("Invalid context - missing required fields:", {
            filepath: !!context?.filepath,
            range: !!context?.range,
          });
          return [];
        }

        const workspaceRoot = context.filepath.split("/")[0];
        if (!workspaceRoot) {
          this.logger.info("Could not determine workspace root from filepath:", context.filepath);
          return [];
        }

        const document = await openTextDocument({ filePath: context.filepath }, this.gitProvider);
        if (!document) {
          this.logger.info(`File not found: ${context.filepath}`);
          return [];
        }

        const textRange = new Range(new Position(context.range.start, 0), new Position(context.range.end + 1, 0));

        const text = document.getText(textRange);
        if (!text) {
          this.logger.info("Empty text content for range:", textRange);
          return [];
        }

        let allResults: Context[] = [];
        const words = text.split(/\b/);

        for (let i = 0; i < words.length; i++) {
          const word = words[i]?.trim();
          if (!word || word.match(/^\W+$/)) {
            continue;
          }

          let pos = 0;
          let currentLine = context.range.start;
          let currentChar = 0;

          for (let j = 0; j < i; j++) {
            const word = words[j];
            if (word) {
              pos += word.length;
            }
          }

          const lines = text.slice(0, pos).split("\n");
          currentLine += lines.length - 1;
          if (lines.length > 1) {
            currentChar = lines[lines.length - 1]?.length ?? 0;
          } else {
            currentChar += pos;
          }

          try {
            const position = new Position(currentLine, currentChar);
            const locations = await getDefinitionLocations(document.uri, position);

            if (!locations?.[0]) {
              continue;
            }

            const location = locations[0];
            const targetUri = location.targetUri?.toString(true);
            if (!targetUri) {
              this.logger.info("Invalid targetUri for location:", location);
              continue;
            }

            const fullPath = targetUri.replace(/^file:\/\/\//, "");
            const relativePath = fullPath.split(`/${workspaceRoot}/`)[1];
            if (!relativePath) {
              this.logger.info("Could not extract relative path from:", { targetUri, workspaceRoot });
              continue;
            }

            const targetDocument = await openTextDocument(
              { filePath: `${workspaceRoot}/${relativePath}` },
              this.gitProvider,
            );
            if (!targetDocument) {
              this.logger.info(`Could not open target file: ${relativePath}`);
              continue;
            }

            if (!location.targetRange?.start || !location.targetRange?.end) {
              this.logger.info("Invalid target range in location:", location);
              continue;
            }

            const targetRange = new Range(
              new Position(location.targetRange.start.line, 0),
              new Position(location.targetRange.end.line + 1, 0),
            );

            const targetContent = targetDocument.getText(targetRange);
            if (!targetContent) {
              this.logger.info("Failed to get target content for range:", targetRange);
              continue;
            }

            allResults.push({
              kind: "file",
              filepath: `${workspaceRoot}/${relativePath}`,
              range: {
                start: location.targetRange.start.line,
                end: location.targetRange.end.line,
              },
              content: targetContent,
              git_url: context.git_url,
            });
          } catch (error) {
            this.logger.error(`Error looking up definition for word "${word}":`, error);
          }
        }

        // transform results
        // remove start 0 and end 0
        allResults = allResults.filter((result) => result.range?.start !== 0 && result.range?.end !== 0);

        // merge overlapping ranges
        const fileGroups = new Map<string, Context[]>();
        for (const result of allResults) {
          if (!result?.filepath) continue;
          const existing = fileGroups.get(result.filepath) || [];
          existing.push(result);
          fileGroups.set(result.filepath, existing);
        }

        const finalResults: Context[] = [];
        for (const [filepath, contexts] of fileGroups) {
          if (!contexts?.length) continue;

          if (contexts.length === 1) {
            const context = contexts[0];
            if (context) {
              finalResults.push(context);
            }
            continue;
          }

          contexts.sort((a, b) => (a.range?.start ?? 0) - (b.range?.start ?? 0));
          let current = contexts[0];

          // TODO(Sma1lboy): consider move range.ts to common packages
          // for handle all case including Range from vscode and vscode-languageserver, also our own define LineRange
          for (let i = 1; i < contexts.length; i++) {
            const next = contexts[i];
            if (!next?.range?.start || !next?.range?.end || !current?.range?.end || !current?.range?.start) {
              this.logger.info("Invalid range:", { current, next });
              continue;
            }

            if (next.range.start <= current.range.end + 1) {
              current = {
                ...current,
                range: {
                  start: Math.min(current.range.start, next.range.start),
                  end: Math.max(current.range.end, next.range.end),
                },
              };

              const targetDocument = await openTextDocument({ filePath: filepath }, this.gitProvider);
              if (targetDocument) {
                const mergedRange = new Range(
                  new Position(current.range.start, 0),
                  new Position(current.range.end + 1, 0),
                );
                const mergedContent = targetDocument.getText(mergedRange);
                if (mergedContent) {
                  current.content = mergedContent;
                }
              }
            } else {
              finalResults.push(current);
              current = next;
            }
          }

          if (current) {
            finalResults.push(current);
          }
        }

        return finalResults;
      },
    });
  }
}
