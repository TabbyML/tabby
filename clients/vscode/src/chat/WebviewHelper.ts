
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
}