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
} from "vscode";
import type { ServerApi, ChatMessage, Context, NavigateOpts, FocusKeybinding } from "tabby-chat-panel";{}
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
import { throws } from "assert";
export class ChatSideViewProvider implements WebviewViewProvider {
  webview?: WebviewView;
  client?: ServerApi;
  private webviewHelper : WebviewHelper;

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
    return WebviewHelper.getFileContextFromSelection({ editor, gitProvider });
  }

  static getFileContextFromEditor({ editor, gitProvider }: { editor: TextEditor; gitProvider: GitProvider }): Context {
    const content = editor.document.getText();
    const lineCount = editor.document.lineCount;
    const uri = editor.document.uri;
    const { filepath, git_url } = WebviewHelper.resolveFilePathAndGitUrl(uri, gitProvider);
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
  public async resolveWebviewView(webviewView: WebviewView) {
    this.webview = webviewView;
    const extensionUri = this.context.extensionUri;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [extensionUri],
    };
    this.webviewHelper.setWebview(webviewView.webview);

    this.client = this.webviewHelper.createChatClient(webviewView.webview);
    this.webviewHelper.setClient(this.client);

    await this.webviewHelper.displayPageBasedOnServerStatus();    
    this.webviewHelper.addAgentEventListeners();

    // The event will not be triggered during the initial rendering.
    webviewView.onDidChangeVisibility(() => {
      if (webviewView.visible) {
        this.webviewHelper.refreshChatPage();
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

  public getWebview() {
    return this.webview;
  }

  public sendMessage(message: ChatMessage) {
    this.webviewHelper.sendMessage(message);
  }

  public addRelevantContext(context: Context) {
    this.webviewHelper.addRelevantContext(context);
  }
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
