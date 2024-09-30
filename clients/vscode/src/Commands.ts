import {
  workspace,
  window,
  env,
  commands,
  ExtensionContext,
  CancellationTokenSource,
  Uri,
  Disposable,
  InputBoxValidationSeverity,
  ProgressLocation,
  ThemeIcon,
  QuickPickItem,
  ViewColumn,
} from "vscode";
import os from "os";
import path from "path";
import { strict as assert } from "assert";
import { Client } from "./lsp/Client";
import { Config, PastServerConfig } from "./Config";
import { ContextVariables } from "./ContextVariables";
import { InlineCompletionProvider } from "./InlineCompletionProvider";
import { ChatSideViewProvider } from "./chat/ChatSideViewProvider";
import { ChatPanelViewProvider } from "./chat/ChatPanelViewProvider";
import { GitProvider, Repository } from "./git/GitProvider";
import CommandPalette from "./CommandPalette";
import { showOutputPanel } from "./logger";
import { Issues } from "./Issues";
import { InlineEditController } from "./inline-edit";

export class Commands {
  private chatEditCancellationTokenSource: CancellationTokenSource | null = null;

  constructor(
    private readonly context: ExtensionContext,
    private readonly client: Client,
    private readonly config: Config,
    private readonly issues: Issues,
    private readonly contextVariables: ContextVariables,
    private readonly inlineCompletionProvider: InlineCompletionProvider,
    private readonly chatViewProvider: ChatSideViewProvider,
    private readonly gitProvider: GitProvider,
  ) {
    const registrations = Object.keys(this.commands).map((key) => {
      const commandName = `tabby.${key}`;
      const commandHandler = this.commands[key];
      if (commandHandler) {
        return commands.registerCommand(commandName, commandHandler, this);
      }
      return null;
    });
    const notNullRegistrations = registrations.filter((disposable): disposable is Disposable => disposable !== null);
    this.context.subscriptions.push(...notNullRegistrations);
  }

  private sendMessageToChatPanel(msg: string) {
    const editor = window.activeTextEditor;
    if (editor) {
      commands.executeCommand("tabby.chatView.focus");
      const fileContext = ChatSideViewProvider.getFileContextFromSelection({ editor, gitProvider: this.gitProvider });
      if (!fileContext) {
        window.showInformationMessage("No selected codes");
        return;
      }

      this.chatViewProvider.sendMessage({
        message: msg,
        selectContext: fileContext,
      });
    } else {
      window.showInformationMessage("No active editor");
    }
  }

  private addRelevantContext() {
    const editor = window.activeTextEditor;
    if (!editor) {
      window.showInformationMessage("No active editor");
      return;
    }

    const addContext = () => {
      const fileContext = ChatSideViewProvider.getFileContextFromSelection({ editor, gitProvider: this.gitProvider });
      if (fileContext) {
        this.chatViewProvider.addRelevantContext(fileContext);
      }
    };

    commands.executeCommand("tabby.chatView.focus").then(addContext);
  }

  commands: Record<string, (...args: never[]) => void> = {
    applyCallback: (callback: (() => void) | undefined) => {
      callback?.();
    },
    toggleInlineCompletionTriggerMode: (value: "automatic" | "manual" | undefined) => {
      let target = value;
      if (!target) {
        if (this.config.inlineCompletionTriggerMode === "automatic") {
          target = "manual";
        } else {
          target = "automatic";
        }
      }
      this.config.inlineCompletionTriggerMode = target;
    },
    setApiEndpoint: () => {
      window
        .showInputBox({
          prompt: "Enter the URL of your Tabby Server",
          value: this.config.serverEndpoint,
          validateInput: (input: string) => {
            try {
              const url = new URL(input);
              assert(url.protocol == "http:" || url.protocol == "https:");
            } catch (error) {
              return {
                message: "Please enter a validate http or https URL.",
                severity: InputBoxValidationSeverity.Error,
              };
            }
            return null;
          },
        })
        .then((url) => {
          if (url) {
            this.config.serverEndpoint = url;
          }
        });
    },
    setApiToken: () => {
      const currentToken = this.config.serverToken;
      window
        .showInputBox({
          prompt: "Enter your personal token",
          value: currentToken.length > 0 ? currentToken : undefined,
          password: true,
        })
        .then((token) => {
          if (token === undefined) {
            return; // User canceled
          }
          this.config.serverToken = token;
        });
    },
    openSettings: () => {
      commands.executeCommand("workbench.action.openSettings", "@ext:TabbyML.vscode-tabby");
    },
    openTabbyAgentSettings: () => {
      if (env.appHost !== "desktop") {
        window.showWarningMessage("Tabby Agent config file is not supported in browser.", { modal: true });
        return;
      }
      const agentUserConfig = Uri.joinPath(Uri.file(os.homedir()), ".tabby-client", "agent", "config.toml");
      workspace.fs.stat(agentUserConfig).then(
        () => {
          workspace.openTextDocument(agentUserConfig).then((document) => {
            window.showTextDocument(document);
          });
        },
        () => {
          window.showWarningMessage("Failed to open Tabby Agent config file.", { modal: true });
        },
      );
    },
    openOnlineHelp: (path?: string | undefined) => {
      if (typeof path === "string" && path.length > 0) {
        env.openExternal(Uri.parse(`https://tabby.tabbyml.com${path}`));
        return;
      }
      window
        .showQuickPick([
          {
            label: "Website",
            iconPath: new ThemeIcon("book"),
            alwaysShow: true,
            description: "Visit Tabby's website to learn more about features and use cases",
          },
          {
            label: "Tabby Slack Community",
            description: "Join Tabby's Slack community to get help or share feedback",
            iconPath: new ThemeIcon("comment-discussion"),
            alwaysShow: true,
          },
          {
            label: "Tabby GitHub Repository",
            description: "Open issues for bugs or feature requests",
            iconPath: new ThemeIcon("github"),
            alwaysShow: true,
          },
        ])
        .then((selection) => {
          if (selection) {
            switch (selection.label) {
              case "Website":
                env.openExternal(Uri.parse("https://tabby.tabbyml.com/"));
                break;
              case "Tabby Slack Community":
                env.openExternal(Uri.parse("https://links.tabbyml.com/join-slack-extensions/"));
                break;
              case "Tabby GitHub Repository":
                env.openExternal(Uri.parse("https://github.com/tabbyml/tabby"));
                break;
            }
          }
        });
    },
    openKeybindings: () => {
      commands.executeCommand("workbench.action.openGlobalKeybindings", "Tabby");
    },
    gettingStarted: () => {
      commands.executeCommand("workbench.action.openWalkthrough", "TabbyML.vscode-tabby#gettingStarted");
    },
    "commandPalette.trigger": () => {
      new CommandPalette(this.client, this.config, this.issues);
    },
    "outputPanel.focus": () => {
      showOutputPanel();
    },
    "inlineCompletion.trigger": () => {
      commands.executeCommand("editor.action.inlineSuggest.trigger");
    },
    "inlineCompletion.accept": () => {
      commands.executeCommand("editor.action.inlineSuggest.commit");
    },
    "inlineCompletion.acceptNextWord": () => {
      this.inlineCompletionProvider.handleEvent("accept_word");
      commands.executeCommand("editor.action.inlineSuggest.acceptNextWord");
    },
    "inlineCompletion.acceptNextLine": () => {
      this.inlineCompletionProvider.handleEvent("accept_line");
      // FIXME: this command move cursor to next line, but we want to move cursor to the end of current line
      commands.executeCommand("editor.action.inlineSuggest.acceptNextLine");
    },
    "inlineCompletion.dismiss": () => {
      this.inlineCompletionProvider.handleEvent("dismiss");
      commands.executeCommand("editor.action.inlineSuggest.hide");
    },
    "notifications.mute": (type: string) => {
      const notifications = this.config.mutedNotifications;
      if (!notifications.includes(type)) {
        const updated = notifications.concat(type);
        this.config.mutedNotifications = updated;
      }
    },
    "notifications.resetMuted": (type?: string) => {
      const notifications = this.config.mutedNotifications;
      if (type) {
        const updated = notifications.filter((item) => item !== type);
        this.config.mutedNotifications = updated;
      } else {
        this.config.mutedNotifications = [];
      }
    },
    "chat.explainCodeBlock": async (userCommand?: string) => {
      this.sendMessageToChatPanel("Explain the selected code:".concat(userCommand ? `\n${userCommand}` : ""));
    },
    "chat.addRelevantContext": async () => {
      this.addRelevantContext();
    },
    "chat.addFileContext": () => {
      const editor = window.activeTextEditor;
      if (editor) {
        commands.executeCommand("tabby.chatView.focus").then(() => {
          const fileContext = ChatSideViewProvider.getFileContextFromEditor({ editor, gitProvider: this.gitProvider });
          this.chatViewProvider.addRelevantContext(fileContext);
        });
      } else {
        window.showInformationMessage("No active editor");
      }
    },
    "chat.fixCodeBlock": async () => {
      this.sendMessageToChatPanel("Identify and fix potential bugs in the selected code:");
    },
    "chat.generateCodeBlockDoc": async () => {
      this.sendMessageToChatPanel("Generate documentation for the selected code:");
    },
    "chat.generateCodeBlockTest": async () => {
      this.sendMessageToChatPanel("Generate a unit test for the selected code:");
    },
    "chat.createPanel": async () => {
      const panel = window.createWebviewPanel("tabby.chatView", "Tabby", ViewColumn.One, {
        retainContextWhenHidden: true,
      });

      const chatPanelViewProvider = new ChatPanelViewProvider(this.context, this.client.agent, this.gitProvider);

      chatPanelViewProvider.resolveWebviewView(panel);
    },
    "chat.edit.start": async (userCommand?: string) => {
      const editor = window.activeTextEditor;
      if (!editor) {
        return;
      }

      const editLocation = {
        uri: editor.document.uri.toString(),
        range: {
          start: { line: editor.selection.start.line, character: 0 },
          end: {
            line: editor.selection.end.character === 0 ? editor.selection.end.line : editor.selection.end.line + 1,
            character: 0,
          },
        },
      };

      const inlineEditController = new InlineEditController(
        this.client,
        this.config,
        this.contextVariables,
        editor,
        editLocation,
        userCommand,
      );
      inlineEditController.start();
    },
    "chat.edit.stop": async () => {
      this.chatEditCancellationTokenSource?.cancel();
    },
    "chat.edit.accept": async () => {
      const editor = window.activeTextEditor;
      if (!editor) {
        return;
      }
      const location = {
        uri: editor.document.uri.toString(),
        range: {
          start: { line: editor.selection.start.line, character: 0 },
          end: { line: editor.selection.end.line + 1, character: 0 },
        },
      };
      await this.client.chat.resolveEdit({ location, action: "accept" });
    },
    "chat.edit.discard": async () => {
      const editor = window.activeTextEditor;
      if (!editor) {
        return;
      }
      const location = {
        uri: editor.document.uri.toString(),
        range: {
          start: { line: editor.selection.start.line, character: 0 },
          end: { line: editor.selection.end.line + 1, character: 0 },
        },
      };
      await this.client.chat.resolveEdit({ location, action: "discard" });
    },
    "chat.generateCommitMessage": async (repository?: Repository) => {
      const repos = this.gitProvider.getRepositories() ?? [];
      if (repos.length < 1) {
        window.showInformationMessage("No Git repositories found.");
        return;
      }
      let selectedRepo = repository;
      if (!selectedRepo) {
        if (repos.length == 1) {
          selectedRepo = repos[0];
        } else {
          const selected = await window.showQuickPick(
            repos
              .map((repo) => {
                const repoRoot = repo.rootUri.fsPath;
                return {
                  label: path.basename(repoRoot),
                  detail: repoRoot,
                  iconPath: new ThemeIcon("repo"),
                  picked: repo.ui.selected,
                  alwaysShow: true,
                  value: repo,
                };
              })
              .sort((a, b) => {
                if (a.detail.startsWith(b.detail)) {
                  return 1;
                } else if (b.detail.startsWith(a.detail)) {
                  return -1;
                } else {
                  return a.label.localeCompare(b.label);
                }
              }),
            { placeHolder: "Select a Git repository" },
          );
          selectedRepo = selected?.value;
        }
      }
      if (!selectedRepo) {
        return;
      }
      window.withProgress(
        {
          location: ProgressLocation.Notification,
          title: "Generating commit message...",
          cancellable: true,
        },
        async (_, token) => {
          // Focus on scm view
          commands.executeCommand("workbench.view.scm");
          const result = await this.client.chat.generateCommitMessage(
            { repository: selectedRepo.rootUri.toString() },
            token,
          );
          if (result && selectedRepo.inputBox) {
            selectedRepo.inputBox.value = result.commitMessage;
          }
        },
      );
    },
    "server.selectPastServerConfig": () => {
      const configs = this.config.pastServerConfigs;
      if (configs.length <= 0) return;

      const quickPick = window.createQuickPick<QuickPickItem & PastServerConfig>();

      quickPick.items = configs.map((x) => ({
        ...x,
        label: x.endpoint,
        buttons: [
          {
            iconPath: new ThemeIcon("settings-remove"),
          },
        ],
      }));

      quickPick.onDidAccept(() => {
        const item = quickPick.activeItems[0];
        if (item) {
          this.config.restoreServerConfig(item);
        }

        quickPick.hide();
      });

      quickPick.onDidTriggerItemButton((e) => {
        if (!(e.button.iconPath instanceof ThemeIcon)) return;
        if (e.button.iconPath.id === "settings-remove") {
          this.config.removePastServerConfigByApiEndpoint(e.item.endpoint);
        }
      });

      quickPick.show();
    },
  };
}
