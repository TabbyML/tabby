import type { Location } from "vscode-languageclient";
import {
  CancellationTokenSource,
  QuickPickItem,
  ThemeIcon,
  QuickPickItemKind,
  window,
  TextEditor,
  Selection,
  Position,
  CancellationToken,
  Range,
  workspace,
  Uri,
  TabInputText,
} from "vscode";
import { ChatEditCommand, FileContext } from "tabby-agent";
import { Client } from "../lsp/client";
import { Config } from "../Config";
import { ContextVariables } from "../ContextVariables";
import { getLogger } from "../logger";
import { parseInput, InlineChatParseResult, replaceLastOccurrence } from "./util";
import { caseInsensitivePattern, findFiles } from "../findFiles";

export class InlineEditController {
  private readonly logger = getLogger("InlineEditController");
  private readonly editLocation: Location;
  private maxSearchFileResult = 30;

  constructor(
    private client: Client,
    private config: Config,
    private contextVariables: ContextVariables,
    private editor: TextEditor,
    private range: Range,
  ) {
    this.editLocation = {
      uri: this.editor.document.uri.toString(),
      range: {
        start: { line: this.range.start.line, character: 0 },
        end: {
          line: this.range.end.character === 0 ? this.range.end.line : this.range.end.line + 1,
          character: 0,
        },
      },
    };
  }

  private get workspaceFolder() {
    return workspace.getWorkspaceFolder(this.editor.document.uri);
  }

  async searchFileList(searchQuery: string) {
    if (!this.workspaceFolder) {
      return [];
    }
    const globPattern = caseInsensitivePattern(searchQuery);
    const fileList = await findFiles(globPattern, { maxResults: this.maxSearchFileResult });
    return fileList.sort((a, b) => a.fsPath.length - b.fsPath.length).map((file) => workspace.asRelativePath(file));
  }

  async start(userCommand: string | undefined, cancellationToken: CancellationToken) {
    const input: InlineChatParseResult | undefined = userCommand
      ? { command: userCommand }
      : await this.showQuickPick();
    if (input?.command) {
      await this.provideEditWithCommand(input, cancellationToken);
    }
  }

  private getFileContext(mentions?: string[]): FileContext[] | undefined {
    if (!mentions) {
      return undefined;
    }
    const worksapceUri = this.workspaceFolder?.uri;
    if (!worksapceUri) {
      return;
    }
    return mentions.map<FileContext>((mention) => ({
      path: Uri.joinPath(worksapceUri, mention).fsPath,
    }));
  }

  private async showQuickPick(): Promise<InlineChatParseResult | undefined> {
    return new Promise((resolve) => {
      const quickPick = window.createQuickPick<CommandQuickPickItem>();
      quickPick.placeholder = "Enter the command for editing (type @ to include file)";
      quickPick.matchOnDescription = true;

      const recentlyCommand = this.config.chatEditRecentlyCommand.slice(0, this.config.maxChatEditHistory);
      const suggestedCommand: ChatEditCommand[] = [];

      const getInputParseResult = (): InlineChatParseResult => {
        return parseInput(quickPick.value);
      };

      const getCommandList = (input: string) => {
        const list: (QuickPickItem & { value: string })[] = [];
        list.push(
          ...suggestedCommand.map((item) => ({
            label: item.label,
            value: item.command,
            iconPath: item.source === "preset" ? new ThemeIcon("run") : new ThemeIcon("spark"),
            description: item.source === "preset" ? item.command : "Suggested",
            alwaysShow: true,
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
        const recentlyCommandToAdd = recentlyCommand.filter(
          (item) => !list.find((i) => i.value === item) && item.includes(input),
        );
        list.push(
          ...recentlyCommandToAdd.map((item) => ({
            label: item,
            value: item,
            iconPath: new ThemeIcon("history"),
            description: "History",
            alwaysShow: true,
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

        return list;
      };

      const getFileListFromMention = async (mention: string) => {
        const list: (QuickPickItem & { value: string })[] = [];
        if (mention === "") {
          if (window.activeTextEditor) {
            const path = workspace.asRelativePath(window.activeTextEditor?.document.uri);
            list.push({
              label: path,
              value: path,
              alwaysShow: true,
            });
          }

          if (list.length > 0) {
            list.push({
              label: "",
              value: "",
              kind: QuickPickItemKind.Separator,
              alwaysShow: true,
            });
          }

          getOpenTabs().forEach((path) => {
            if (list.find((i) => i.value === path) === undefined) {
              list.push({
                label: path,
                value: path,
                alwaysShow: true,
              });
            }
          });
        } else {
          const fileList = await this.searchFileList(mention);
          list.push(
            ...fileList.map((file) => ({
              label: file,
              value: file,
              alwaysShow: true,
            })),
          );
        }

        return list;
      };

      const updateQuickPickList = async () => {
        let list: (QuickPickItem & { value: string })[] = [];
        const { mentionQuery, command } = getInputParseResult();
        if (mentionQuery !== undefined) {
          quickPick.busy = true;
          list = await getFileListFromMention(mentionQuery);
          quickPick.busy = false;
        } else {
          list = getCommandList(command);
        }
        quickPick.items = list;
      };

      quickPick.onDidChangeValue(() => updateQuickPickList());

      const fetchingSuggestedCommandCancellationTokenSource = new CancellationTokenSource();
      this.client.chat.provideEditCommands(
        { location: this.editLocation },
        { commands: suggestedCommand, callback: () => updateQuickPickList() },
        fetchingSuggestedCommandCancellationTokenSource.token,
      );

      quickPick.onDidTriggerItemButton(async (event) => {
        const item = event.item;
        const button = event.button;
        if (button.iconPath instanceof ThemeIcon && button.iconPath.id === "settings-remove") {
          const index = recentlyCommand.indexOf(item.value);
          if (index !== -1) {
            recentlyCommand.splice(index, 1);
            await this.config.updateChatEditRecentlyCommand(recentlyCommand);
            updateQuickPickList();
          }
        }

        if (button.iconPath instanceof ThemeIcon && button.iconPath.id === "edit") {
          quickPick.value = item.value;
        }
      });

      const acceptCommand = async (command: string | undefined, mentions?: string[]) => {
        if (!command) {
          resolve(undefined);
          return;
        }
        if (command && command.length > 200) {
          window.showErrorMessage("Command is too long.");
          resolve(undefined);
          return;
        }

        const recentlyCommand = this.config.chatEditRecentlyCommand;
        const updatedRecentlyCommand = [command]
          .concat(recentlyCommand.filter((item) => item !== command))
          .slice(0, this.config.maxChatEditHistory);
        await this.config.updateChatEditRecentlyCommand(updatedRecentlyCommand);

        resolve({ command, mentions });
        quickPick.hide();
      };

      const acceptFile = async (file: string | undefined, query: string) => {
        if (!file) {
          return;
        }
        const newValue = replaceLastOccurrence(quickPick.value, query, file);
        quickPick.value = newValue + " ";
      };

      quickPick.onDidAccept(async () => {
        const commandOrFile = quickPick.selectedItems[0]?.value;
        const inputParseResult = getInputParseResult();
        if (inputParseResult.mentionQuery !== undefined) {
          acceptFile(commandOrFile, inputParseResult.mentionQuery);
        } else {
          acceptCommand(commandOrFile, inputParseResult.mentions);
        }
      });
      quickPick.onDidHide(() => {
        fetchingSuggestedCommandCancellationTokenSource.cancel();
        resolve(undefined);
      });

      quickPick.show();
    });
  }

  private async provideEditWithCommand(inputResult: InlineChatParseResult, cancellationToken: CancellationToken) {
    // Lock the cursor (editor selection) at start position, it will be unlocked after the edit is done
    const startPosition = new Position(this.range.start.line, 0);
    const resetEditorSelection = () => {
      this.editor.selection = new Selection(startPosition, startPosition);
    };
    const selectionListenerDisposable = window.onDidChangeTextEditorSelection((event) => {
      if (event.textEditor === this.editor) {
        resetEditorSelection();
      }
    });
    resetEditorSelection();

    this.contextVariables.chatEditInProgress = true;
    this.logger.log(`Provide edit with command: ${JSON.stringify(inputResult)}`);
    try {
      await this.client.chat.provideEdit(
        {
          location: this.editLocation,
          command: inputResult.command,
          context: this.getFileContext(inputResult.mentions),
          format: "previewChanges",
        },
        cancellationToken,
      );
    } catch (error) {
      if (typeof error === "object" && error && "message" in error && typeof error["message"] === "string") {
        if (cancellationToken.isCancellationRequested || error["message"].includes("This operation was aborted")) {
          // user canceled
        } else {
          window.showErrorMessage(error["message"]);
        }
      }
    }
    selectionListenerDisposable.dispose();
    this.contextVariables.chatEditInProgress = false;
  }
}

interface CommandQuickPickItem extends QuickPickItem {
  value: string;
}

export const getOpenTabs = () => {
  return window.tabGroups.all
    .flatMap((group) =>
      group.tabs.map((tab) => (tab.input instanceof TabInputText ? workspace.asRelativePath(tab.input.uri) : null)),
    )
    .filter((item): item is string => item !== null);
};
