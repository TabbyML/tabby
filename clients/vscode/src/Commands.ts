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
  Location,
  Range,
  TextDocument,
} from "vscode";
import os from "os";
import path from "path";
import { strict as assert } from "assert";
import { ChatEditCommand } from "tabby-agent";
import { Client } from "./lsp/Client";
import { Config, PastServerConfig } from "./Config";
import { ContextVariables } from "./ContextVariables";
import { InlineCompletionProvider } from "./InlineCompletionProvider";
import { ChatViewProvider } from "./chat/ChatViewProvider";
import { GitProvider, Repository } from "./git/GitProvider";
import CommandPalette from "./CommandPalette";
import { getLogger, showOutputPanel } from "./logger";
import { Issues } from "./Issues";
import { NLOutlinesProvider } from "./NLOutlinesProvider";

export class Commands {
  private chatEditCancellationTokenSource: CancellationTokenSource | null = null;
  private nlOutlinesCancellationTokenSource: CancellationTokenSource | null = null;

  constructor(
    private readonly context: ExtensionContext,
    private readonly client: Client,
    private readonly config: Config,
    private readonly issues: Issues,
    private readonly contextVariables: ContextVariables,
    private readonly inlineCompletionProvider: InlineCompletionProvider,
    private readonly chatViewProvider: ChatViewProvider,
    private readonly gitProvider: GitProvider,
    private readonly nlOutlinesProvider: NLOutlinesProvider,
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
      const fileContext = ChatViewProvider.getFileContextFromSelection({ editor, gitProvider: this.gitProvider });
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

    // If chat webview is not created or not visible, we shall focus on it.
    const focusChat = !this.chatViewProvider.webview?.visible;
    const addContext = () => {
      const fileContext = ChatViewProvider.getFileContextFromSelection({ editor, gitProvider: this.gitProvider });
      if (fileContext) {
        this.chatViewProvider.addRelevantContext(fileContext);
      }
    };

    if (focusChat) {
      commands.executeCommand("tabby.chatView.focus").then(addContext);
    } else {
      addContext();
    }
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
    "chat.explainCodeBlock": async () => {
      this.sendMessageToChatPanel("Explain the selected code:");
    },
    "chat.addRelevantContext": async () => {
      this.addRelevantContext();
    },
    "chat.addFileContext": () => {
      const editor = window.activeTextEditor;
      if (editor) {
        commands.executeCommand("tabby.chatView.focus").then(() => {
          const fileContext = ChatViewProvider.getFileContextFromEditor({ editor, gitProvider: this.gitProvider });
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
      //ensure max length
      const recentlyCommand = this.config.chatEditRecentlyCommand.slice(0, this.config.maxChatEditHistory);
      const suggestedCommand: ChatEditCommand[] = [];
      const quickPick = window.createQuickPick<QuickPickItem & { value: string }>();

      const updateQuickPickList = () => {
        const input = quickPick.value;
        const list: (QuickPickItem & { value: string })[] = [];
        list.push(
          ...suggestedCommand.map((item) => ({
            label: item.label,
            value: item.command,
            iconPath: item.source === "preset" ? new ThemeIcon("run") : new ThemeIcon("spark"),
            description: item.source === "preset" ? item.command : "Suggested",
          })),
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
          ...recentlyCommandToAdd.map((item) => ({
            label: item,
            value: item,
            iconPath: new ThemeIcon("history"),
            description: "History",
            buttons: [
              {
                iconPath: new ThemeIcon("edit"),
              },
              {
                iconPath: new ThemeIcon("settings-remove"),
              },
            ],
          })),
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
          const updatedRecentlyCommand = [command]
            .concat(recentlyCommand.filter((item) => item !== command))
            .slice(0, this.config.maxChatEditHistory);
          this.config.chatEditRecentlyCommand = updatedRecentlyCommand;

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

      quickPick.onDidTriggerItemButton((event) => {
        const item = event.item;
        const button = event.button;
        if (button.iconPath instanceof ThemeIcon && button.iconPath.id === "settings-remove") {
          const index = recentlyCommand.indexOf(item.value);
          if (index !== -1) {
            recentlyCommand.splice(index, 1);
            this.config.chatEditRecentlyCommand = recentlyCommand;
            updateQuickPickList();
          }
        }

        if (button.iconPath instanceof ThemeIcon && button.iconPath.id === "edit") {
          quickPick.value = item.value;
        }
      });

      quickPick.show();
    },
    "chat.edit.generateNLOutlines": async () => {
      const editor = window.activeTextEditor;
      if (!editor) {
        return;
      }

      const getOffsetRange = (document: TextDocument, start: number, end: number, offset: number): Range => {
        const offsetStart = Math.max(0, start - offset);
        const offsetEnd = Math.min(document.lineCount - 1, end + offset);
        return new Range(new Position(offsetStart, 0), document.lineAt(offsetEnd).range.end);
      };

      let editLocation: Location;
      if (editor.selection.isEmpty) {
        const visibleRanges = editor.visibleRanges;
        if (visibleRanges.length > 0) {
          const firstVisibleLine = visibleRanges[0]?.start.line;
          const lastVisibleLine = visibleRanges[visibleRanges.length - 1]?.end.line;
          if (firstVisibleLine === undefined || lastVisibleLine === undefined) {
            return;
          }
          const offsetRange = getOffsetRange(editor.document, firstVisibleLine, lastVisibleLine, 20);
          editLocation = {
            uri: editor.document.uri,
            range: offsetRange,
          };
        } else {
          const currentLine = editor.selection.active.line;
          const offsetRange = getOffsetRange(editor.document, currentLine, currentLine, 20);
          editLocation = {
            uri: editor.document.uri,
            range: offsetRange,
          };
        }
      } else {
        editLocation = {
          uri: editor.document.uri,
          range: new Range(editor.selection.start, editor.selection.end),
        };
      }

      window.withProgress(
        {
          location: ProgressLocation.Notification,
          title: "Generating natural language outlines...",
          cancellable: true,
        },
        async (_, token) => {
          this.contextVariables.nlOutlinesGenerationInProgress = true;
          if (token.isCancellationRequested) {
            return;
          }
          this.nlOutlinesCancellationTokenSource = new CancellationTokenSource();
          token.onCancellationRequested(() => {
            this.nlOutlinesCancellationTokenSource?.cancel();
          });
          try {
            await this.nlOutlinesProvider.provideNLOutlinesGenerate({
              location: editLocation,
              editor: editor,
            });
          } catch (error) {
            if (typeof error === "object" && error && "message" in error && typeof error.message === "string") {
              window.showErrorMessage(`Error generating outlines: ${error.message}`);
            }
          } finally {
            this.nlOutlinesCancellationTokenSource?.dispose();
            this.nlOutlinesCancellationTokenSource = null;
            this.contextVariables.nlOutlinesGenerationInProgress = false;
          }
        },
      );
    },
    "chat.edit.editNLOutline": async (uri?: Uri, startLine?: number) => {
      const editor = window.activeTextEditor;
      if (!editor) return;
      let documentUri: string;
      let line: number;
      if (uri && startLine !== undefined) {
        documentUri = uri.toString();
        line = startLine;
      } else {
        documentUri = editor.document.uri.toString();
        line = editor.selection.active.line;
      }
      const content = this.nlOutlinesProvider.getOutline(documentUri, line);
      getLogger().info("get content");
      if (!content) return;
      getLogger().info("shown");
      const quickPick = window.createQuickPick();
      quickPick.items = [{ label: content }];
      quickPick.placeholder = "Edit NL Outline content";
      quickPick.value = content;
      quickPick.onDidAccept(async () => {
        const newContent = quickPick.value;
        quickPick.hide();

        await window.withProgress(
          {
            location: ProgressLocation.Notification,
            title: "Updating NL Outline",
            cancellable: false,
          },
          async (progress) => {
            progress.report({ increment: 0 });

            try {
              await this.nlOutlinesProvider.updateNLOutline(documentUri, line, newContent);
              progress.report({ increment: 100 });
              window.showInformationMessage(`Updated NL Outline: ${newContent}`);
            } catch (error) {
              getLogger().error("Error updating NL Outline:", error);
              window.showErrorMessage(
                `Error updating NL Outline: ${error instanceof Error ? error.message : String(error)}`,
              );
            }
          },
        );
      });
      quickPick.show();
    },
    "chat.edit.outline.accept": async () => {
      const editor = window.activeTextEditor;
      if (!editor) {
        return;
      }
      // const location = {
      //   uri: editor.document.uri.toString(),
      //   range: {
      //     start: { line: editor.selection.start.line, character: 0 },
      //     end: { line: editor.selection.end.line + 1, character: 0 },
      //   },
      // };
      await this.nlOutlinesProvider.resolveOutline("accept");
    },
    "chat.edit.outline.discard": async () => {
      const editor = window.activeTextEditor;
      if (!editor) {
        return;
      }
      // const location = {
      //   uri: editor.document.uri.toString(),
      //   range: {
      //     start: { line: editor.selection.start.line, character: 0 },
      //     end: { line: editor.selection.end.line + 1, character: 0 },
      //   },
      // };
      await this.nlOutlinesProvider.resolveOutline("discard");
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
