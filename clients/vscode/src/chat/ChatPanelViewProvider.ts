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
} from "vscode";
import type { ServerApi, ChatMessage, Context, NavigateOpts, FocusKeybinding } from "tabby-chat-panel";
import { WebviewHelper } from "./WebviewHelper";
import hashObject from "object-hash";
import * as semver from "semver";
import type { ServerInfo } from "tabby-agent";
import type { AgentFeature as Agent } from "../lsp/AgentFeature";
import { createClient } from "./chatPanel";
import { GitProvider } from "../git/GitProvider";
import { getLogger } from "../logger";
import { contributes } from "../../package.json";
import { parseKeybinding, readUserKeybindingsConfig } from "../util/KeybindingParser";
// TODO(zhizhg): abstruct a base class with ChatSideViewProvider
export class ChatPanelViewProvider {
  webview?: WebviewPanel;
  client?: ServerApi;
  private webviewHelper : WebviewHelper;
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
  ) {
    this.webviewHelper = new WebviewHelper(context, agent, logger, gitProvider);
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

    const { filepath, git_url } = resolveFilePathAndGitUrl(uri, gitProvider);

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

  static getFileContextFromEditor({ editor, gitProvider }: { editor: TextEditor; gitProvider: GitProvider }): Context {
    const content = editor.document.getText();
    const lineCount = editor.document.lineCount;
    const uri = editor.document.uri;
    const { filepath, git_url } = resolveFilePathAndGitUrl(uri, gitProvider);
    return {
      kind: "file",
      content,
      range: {
        start: 1,
        end: lineCount,
      },
      filepath,
      git_url,
    };
  }

  // The method is called when the chat panel first opened
  public async resolveWebviewView(webviewView: WebviewPanel) {
    this.webview = webviewView;
    const extensionUri = this.context.extensionUri;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [extensionUri],
    };
    this.webviewHelper.setWebview(webviewView.webview);

    this.client = createClient(webviewView, {
      navigate: async (context: Context, opts?: NavigateOpts) => {
        if (opts?.openInEditor) {
          const document = await resolveDocument(this.logger, workspace.workspaceFolders, context.filepath);
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
        await this.webviewHelper.displayChatPage(serverInfo.config.endpoint, { force: true });
        return;
      },
      onSubmitMessage: async (msg: string, relevantContext?: Context[]) => {
        const editor = window.activeTextEditor;
        const chatMessage: ChatMessage = {
          message: msg,
          relevantContext: [],
        };
        if (editor) {
          const fileContext = ChatPanelViewProvider.getFileContextFromSelection({ editor, gitProvider: this.gitProvider });
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
      onLoaded: () => {
        setTimeout(() => {
          this.refreshChatPage();
        }, 300);
      },
      onCopy: (content) => {
        env.clipboard.writeText(content);
      },
      focusOnEditor: () => {
        const editor = window.activeTextEditor;
        if (editor) {
          getLogger().info("Focus back to active editor");
          commands.executeCommand("workbench.action.focusFirstEditorGroup");
        }
      },
    });

    // At this point, if the server instance is not set up, agent.status is 'notInitialized'.
    // We check for the presence of the server instance by verifying serverInfo.health["webserver"].
    const serverInfo = await this.agent.fetchServerInfo();
    if (serverInfo.health && serverInfo.health["webserver"]) {
      const serverInfo = await this.agent.fetchServerInfo();
      this.webviewHelper.displayChatPage(serverInfo.config.endpoint);
    } else {
      this.webviewHelper.displayDisconnectedPage();
    }

    this.agent.on("didChangeStatus", async (status) => {
      if (status !== "disconnected") {
        const serverInfo = await this.agent.fetchServerInfo();
        this.webviewHelper.displayChatPage(serverInfo.config.endpoint);
        this.refreshChatPage();
      } else if (this.isChatPageDisplayed) {
        this.webviewHelper.displayDisconnectedPage();
      }
    });

    this.agent.on("didUpdateServerInfo", async () => {
      const serverInfo = await this.agent.fetchServerInfo();
      this.webviewHelper.displayChatPage(serverInfo.config.endpoint, { force: true });
      this.refreshChatPage();
    });

    // The event will not be triggered during the initial rendering.
    webviewView.onDidChangeViewState(() => {
      if (webviewView.visible) {
        this.refreshChatPage();
      }

      commands.executeCommand("setContext", "tabby.chatViewVisible", webviewView.visible);
    });

    webviewView.webview.onDidReceiveMessage((message) => {
      switch (message.action) {
        case "sync-theme": {
          this.client?.updateTheme(message.style, getColorThemeString(window.activeColorTheme.kind));
          return;
        }
      }
    });
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

    const error = this.webviewHelper.checkChatPanel(serverInfo);
    if (error) {
      this.client?.showError({ content: error });
      return;
    }

    this.pendingRelevantContexts.forEach((ctx) => this.addRelevantContext(ctx));
    this.pendingMessages.forEach((message) => this.sendMessageToChatPanel(message));

    if (serverInfo.config.token) {
      this.client?.cleanError();

      const focusKeybinding = await this.webviewHelper.getFocusKeybinding();
      getLogger().info("focus key binding: ", focusKeybinding);

      this.client?.init({
        fetcherOptions: {
          authorization: serverInfo.config.token,
        },
        focusKey: focusKeybinding,
      });
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

  public addRelevantContext(context: Context) {
    if (!this.client) {
      this.pendingRelevantContexts.push(context);
    } else {
      this.client?.addRelevantContext(context);
    }
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

function resolveFilePathAndGitUrl(uri: Uri, gitProvider: GitProvider): { filepath: string; git_url: string } {
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

async function resolveDocument(
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

function getColorThemeString(kind: ColorThemeKind) {
  switch (kind) {
    case ColorThemeKind.Light:
    case ColorThemeKind.HighContrastLight:
      return "light";
    case ColorThemeKind.Dark:
    case ColorThemeKind.HighContrast:
      return "dark";
  }
}
