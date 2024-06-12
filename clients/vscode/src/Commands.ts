import {
  workspace,
  window,
  env,
  commands,
  ExtensionContext,
  CancellationTokenSource,
  Uri,
  Position,
  Selection,
  Disposable,
  InputBoxValidationSeverity,
  ProgressLocation,
  ThemeIcon,
  QuickPickItem,
  QuickPickItemKind,
} from "vscode";
import os from "os";
import path from "path";
import { strict as assert } from "assert";
import { ChatEditCommand } from "tabby-agent";
import { Client } from "./lsp/Client";
import { Config } from "./Config";
import { ContextVariables } from "./ContextVariables";
import { InlineCompletionProvider } from "./InlineCompletionProvider";
import { ChatViewProvider } from "./chat/ChatViewProvider";
import { GitProvider, Repository } from "./git/GitProvider";

export class Commands {
  private chatEditCancellationTokenSource: CancellationTokenSource | null = null;

  constructor(
    private readonly context: ExtensionContext,
    private readonly client: Client,
    private readonly config: Config,
    private readonly contextVariables: ContextVariables,
    private readonly inlineCompletionProvider: InlineCompletionProvider,
    private readonly chatViewProvider: ChatViewProvider,
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
            label: "Online Documentation",
            iconPath: new ThemeIcon("book"),
            alwaysShow: true,
          },
          {
            label: "Model Registry",
            description: "Explore more recommend models from Tabby's model registry",
            iconPath: new ThemeIcon("library"),
            alwaysShow: true,
          },
          {
            label: "Tabby Slack Community",
            description: "Join Tabby's Slack community to get help or feed back",
            iconPath: new ThemeIcon("comment-discussion"),
            alwaysShow: true,
          },
          {
            label: "Tabby GitHub Repository",
            description: "View the source code for Tabby, and open issues",
            iconPath: new ThemeIcon("github"),
            alwaysShow: true,
          },
        ])
        .then((selection) => {
          if (selection) {
            switch (selection.label) {
              case "Online Documentation":
                env.openExternal(Uri.parse("https://tabby.tabbyml.com/"));
                break;
              case "Model Registry":
                env.openExternal(Uri.parse("https://tabby.tabbyml.com/docs/models/"));
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
    "chat.explainCodeBlock": async () => {
      const editor = window.activeTextEditor;
      if (editor) {
        commands.executeCommand("tabby.chatView.focus");
        const fileContext = ChatViewProvider.getFileContextFromSelection({ editor, gitProvider: this.gitProvider });
        this.chatViewProvider.sendMessage({
          message: "Explain the selected code:",
          selectContext: fileContext,
        });
      } else {
        window.showInformationMessage("No active editor");
      }
    },
    "chat.fixCodeBlock": async () => {
      const editor = window.activeTextEditor;
      if (editor) {
        commands.executeCommand("tabby.chatView.focus");
        const fileContext = ChatViewProvider.getFileContextFromSelection({ editor, gitProvider: this.gitProvider });
        this.chatViewProvider.sendMessage({
          message: "Identify and fix potential bugs in the selected code:",
          selectContext: fileContext,
        });
      } else {
        window.showInformationMessage("No active editor");
      }
    },
    "chat.generateCodeBlockDoc": async () => {
      const editor = window.activeTextEditor;
      if (editor) {
        commands.executeCommand("tabby.chatView.focus");
        const fileContext = ChatViewProvider.getFileContextFromSelection({ editor, gitProvider: this.gitProvider });
        this.chatViewProvider.sendMessage({
          message: "Generate documentation for the selected code:",
          selectContext: fileContext,
        });
      } else {
        window.showInformationMessage("No active editor");
      }
    },
    "chat.generateCodeBlockTest": async () => {
      const editor = window.activeTextEditor;
      if (editor) {
        commands.executeCommand("tabby.chatView.focus");
        const fileContext = ChatViewProvider.getFileContextFromSelection({ editor, gitProvider: this.gitProvider });
        this.chatViewProvider.sendMessage({
          message: "Generate a unit test for the selected code:",
          selectContext: fileContext,
        });
      } else {
        window.showInformationMessage("No active editor");
      }
    },
    "chat.edit.start": async () => {
      const editor = window.activeTextEditor;
      if (!editor) {
        return;
      }
      const startPosition = new Position(editor.selection.start.line, 0);
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
      const recentlyCommand = this.config.chatEditRecentlyCommand;
      const suggestedCommand: ChatEditCommand[] = [];
      const quickPick = window.createQuickPick<QuickPickItem & { value: string }>();
      const updateQuickPickList = () => {
        const input = quickPick.value;
        const list: (QuickPickItem & { value: string })[] = [];
        list.push(
          ...suggestedCommand.map((item) => {
            return {
              label: item.label,
              value: item.command,
              iconPath: item.source === "preset" ? new ThemeIcon("edit") : new ThemeIcon("spark"),
              description: item.source === "preset" ? item.command : "Suggested",
            };
          }),
        );
        if (list.length > 0) {
          list.push({
            label: "",
            value: "",
            kind: QuickPickItemKind.Separator,
            alwaysShow: true,
          });
        }
        const recentlyCommandToAdd = recentlyCommand.filter((item) => !list.find((i) => i.value === item));
        list.push(
          ...recentlyCommandToAdd.map((item) => {
            return {
              label: item,
              value: item,
              iconPath: new ThemeIcon("history"),
              description: "History",
            };
          }),
        );
        if (input.length > 0 && !list.find((i) => i.value === input)) {
          list.unshift({
            label: input,
            value: input,
            iconPath: new ThemeIcon("run"),
            description: "",
            alwaysShow: true,
          });
        }
        quickPick.items = list;
      };
      const fetchingSuggestedCommandCancellationTokenSource = new CancellationTokenSource();
      this.client.chat.provideEditCommands(
        { location: editLocation },
        { commands: suggestedCommand, callback: () => updateQuickPickList() },
        fetchingSuggestedCommandCancellationTokenSource.token,
      );
      quickPick.placeholder = "Enter the command for editing";
      quickPick.matchOnDescription = true;
      quickPick.onDidChangeValue(() => updateQuickPickList());
      quickPick.onDidHide(() => {
        fetchingSuggestedCommandCancellationTokenSource.cancel();
      });
      quickPick.onDidAccept(() => {
        quickPick.hide();
        const command = quickPick.selectedItems[0]?.value;
        if (command) {
          this.config.chatEditRecentlyCommand = [command]
            .concat(recentlyCommand.filter((item) => item !== command))
            .slice(0, 20);
          window.withProgress(
            {
              location: ProgressLocation.Notification,
              title: "Editing in progress...",
              cancellable: true,
            },
            async (_, token) => {
              editor.selection = new Selection(startPosition, startPosition);
              this.contextVariables.chatEditInProgress = true;
              if (token.isCancellationRequested) {
                return;
              }
              this.chatEditCancellationTokenSource = new CancellationTokenSource();
              token.onCancellationRequested(() => {
                this.chatEditCancellationTokenSource?.cancel();
              });
              try {
                await this.client.chat.provideEdit(
                  {
                    location: editLocation,
                    command,
                    format: "previewChanges",
                  },
                  this.chatEditCancellationTokenSource.token,
                );
              } catch (error) {
                if (typeof error === "object" && error && "message" in error && typeof error["message"] === "string") {
                  window.showErrorMessage(error["message"]);
                }
              }
              this.chatEditCancellationTokenSource.dispose();
              this.chatEditCancellationTokenSource = null;
              this.contextVariables.chatEditInProgress = false;
              editor.selection = new Selection(startPosition, startPosition);
            },
          );
        }
      });
      quickPick.show();
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
    "chat.generateCommitMessage": async () => {
      const repos = this.gitProvider.getRepositories() ?? [];
      if (repos.length < 1) {
        window.showInformationMessage("No Git repositories found.");
        return;
      }
      // Select repo
      let selectedRepo: Repository | undefined = undefined;
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
          if (result) {
            selectedRepo.inputBox.value = result.commitMessage;
          }
        },
      );
    },
  };
}
