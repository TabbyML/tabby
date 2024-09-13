
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
import type { ServerApi, ChatMessage, Context } from "tabby-chat-panel";
import hashObject from "object-hash";
import * as semver from "semver";
import type { ServerInfo } from "tabby-agent";
import type { AgentFeature as Agent } from "../lsp/AgentFeature";
import { GitProvider } from "../git/GitProvider";

export class WebviewHelper {
    webview?: WebviewView | WebviewPanel;
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
}