import { ExtensionContext, WebviewView, workspace, Uri, LogOutputChannel, TextEditor, WebviewPanel } from "vscode";
import type { ServerApi, ChatMessage, Context } from "tabby-chat-panel";
import hashObject from "object-hash";
import * as semver from "semver";
import type { ServerInfo } from "tabby-agent";
import type { AgentFeature as Agent } from "../lsp/AgentFeature";
import { GitProvider } from "../git/GitProvider";

export abstract class BaseChatView {
  webview?: WebviewView | WebviewPanel;
  client?: ServerApi;
  protected pendingMessages: ChatMessage[] = [];
  protected pendingRelevantContexts: Context[] = [];
  protected isChatPageDisplayed = false;

  constructor(
    protected readonly context: ExtensionContext,
    protected readonly agent: Agent,
    protected readonly logger: LogOutputChannel,
    protected readonly gitProvider: GitProvider,
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

  // Check if server is healthy and has the chat model enabled.
  //
  // Returns undefined if it's working, otherwise returns a message to display.
  protected checkChatPanel(serverInfo: ServerInfo): string | undefined {
    if (!serverInfo.health) {
      return "Your Tabby server is not responding. Please check your server status.";
    }

    if (!serverInfo.health["webserver"] || !serverInfo.health["chat_model"]) {
      return "You need to launch the server with the chat model enabled; for example, use `--chat-model Qwen2-1.5B-Instruct`.";
    }

    const MIN_VERSION = "0.16.0";

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

  protected async refreshChatPage() {
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
      // Duplicate init won't break or reload the current chat page
      this.client?.init({
        fetcherOptions: {
          authorization: serverInfo.config.token,
        },
      });
    }
  }

  public addRelevantContext(context: Context) {
    if (!this.client) {
      this.pendingRelevantContexts.push(context);
    } else {
      this.client?.addRelevantContext(context);
    }
  }

  protected sendMessageToChatPanel(message: ChatMessage) {
    this.logger.info(`Sending message to chat panel: ${JSON.stringify(message)}`);
    this.client?.sendMessage(message);
  }

  protected displayDisconnectedPage() {
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

  protected async displayChatPage(endpoint: string, opts?: { force: boolean }) {
    if (!endpoint) return;
    if (this.isChatPageDisplayed && !opts?.force) return;

    if (this.webview) {
      this.isChatPageDisplayed = true;
      const styleUri = this.webview?.webview.asWebviewUri(
        Uri.joinPath(this.context.extensionUri, "assets", "chat-panel.css"),
      );

      const logoUri = this.webview?.webview.asWebviewUri(
        Uri.joinPath(this.context.extensionUri, "assets", "tabby.png"),
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
					const loadingOverlay = document.getElementById("loading-overlay");
	
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
	
						  setTimeout(() => {
							loadingOverlay.style.display = 'none';
							chatIframe.style.display = 'block';
						  }, 0)
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

  protected formatLineHashForCodeBrowser(
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
