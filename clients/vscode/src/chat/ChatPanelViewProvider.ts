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

  // The method is called when the chat panel first opened
  public async resolveWebviewView(webviewView: WebviewPanel) {
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
    webviewView.onDidChangeViewState(() => {
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
