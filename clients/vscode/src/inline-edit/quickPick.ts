import {
  CancellationTokenSource,
  QuickInputButton,
  QuickInputButtons,
  QuickPickItem,
  QuickPickItemButtonEvent,
  QuickPickItemKind,
  TabInputText,
  TextEditor,
  ThemeIcon,
  Uri,
  window,
  workspace,
} from "vscode";
import { Config } from "../Config";
import { ChatEditCommand, ChatEditFileContext } from "tabby-agent";
import { Deferred, InlineEditParseResult, parseUserCommand, replaceLastOccurrence } from "./util";
import { Client } from "../lsp/client";
import { Location } from "vscode-languageclient";
import { caseInsensitivePattern, findFiles } from "../findFiles";
import { wrapCancelableFunction } from "../cancelableFunction";

export interface InlineEditCommand {
  command: string;
  context?: ChatEditFileContext[];
}

interface CommandQuickPickItem extends QuickPickItem {
  value: string;
}

export class UserCommandQuickpick {
  quickPick = window.createQuickPick<CommandQuickPickItem>();
  private suggestedCommand: ChatEditCommand[] = [];
  private resultDeffer = new Deferred<InlineEditCommand | undefined>();
  private fetchingSuggestedCommandCancellationTokenSource = new CancellationTokenSource();
  private lastInputValue = "";
  private filePick: FileSelectionQuickPick | undefined;
  private fileContextLabelToUriMap = new Map<string, string>();

  constructor(
    private client: Client,
    private config: Config,
    private editor: TextEditor,
    private editLocation: Location,
  ) {
    this.editLocation = editLocation;
  }

  start() {
    this.quickPick.title = "Enter the command for editing (type @ to include file)";
    this.quickPick.matchOnDescription = true;
    this.quickPick.onDidChangeValue(() => this.handleValueChange());
    this.quickPick.onDidAccept(() => this.handleAccept());
    this.quickPick.onDidHide(() => this.handleHidden());
    this.quickPick.onDidTriggerItemButton((e) => this.handleTriggerItemButton(e));
    this.quickPick.show();
    this.quickPick.ignoreFocusOut = true;
    this.provideEditCommands();
    return this.resultDeffer.promise;
  }

  private get inputParseResult(): InlineEditParseResult {
    return parseUserCommand(this.quickPick.value);
  }

  private handleValueChange() {
    const { mentionQuery } = this.inputParseResult;
    if (mentionQuery === "") {
      this.openFilePick();
    } else {
      this.updateQuickPickList();
      this.updateQuickPickValue(this.quickPick.value);
    }
  }

  private async openFilePick() {
    this.filePick = new FileSelectionQuickPick(this.editor);
    const file = await this.filePick.start();
    this.quickPick.show();
    if (file) {
      this.updateQuickPickValue(this.quickPick.value + `${file.label} `);
      this.fileContextLabelToUriMap.set(file.label, file.uri);
    } else {
      // remove `@` when user select no file
      this.updateQuickPickValue(replaceLastOccurrence(this.quickPick.value, "@", ""));
    }
    this.filePick = undefined;
  }

  private updateQuickPickValue(value: string) {
    const lastQuickPickValue = this.lastInputValue;
    const lastMentionQuery = parseUserCommand(lastQuickPickValue).mentionQuery;
    const currentMentionQuery = parseUserCommand(value).mentionQuery;
    // remove whole `@file` part when user start delete on the last `@file`
    if (
      lastMentionQuery !== undefined &&
      currentMentionQuery !== undefined &&
      currentMentionQuery.length < lastMentionQuery.length
    ) {
      this.quickPick.value = replaceLastOccurrence(value, `@${currentMentionQuery}`, "");
    } else {
      this.quickPick.value = value;
    }
    this.lastInputValue = this.quickPick.value;
  }

  private async updateQuickPickList() {
    const command = this.quickPick.value;
    const list = this.getCommandList(command);
    this.quickPick.items = list;
  }

  private getCommandList(input: string) {
    const list: (QuickPickItem & { value: string })[] = [];
    list.push(
      ...this.suggestedCommand.map((item) => ({
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
    const recentlyCommandToAdd = this.getCommandHistory().filter((item) => !list.find((i) => i.value === item.command));
    recentlyCommandToAdd.forEach((command) => {
      if (command.context) {
        command.context.forEach((context) => {
          if (!this.fileContextLabelToUriMap.has(context.referrer)) {
            // this context maybe outdated
            this.fileContextLabelToUriMap.set(context.referrer, context.uri);
          }
        });
      }
    });
    list.push(
      ...recentlyCommandToAdd.map((item) => ({
        label: item.command,
        value: item.command,
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

    return list;
  }

  private handleAccept() {
    const command = this.quickPick.selectedItems[0]?.value;
    this.acceptCommand(command);
  }

  private getCommandHistory(): InlineEditCommand[] {
    const recentlyCommand = this.config.chatEditRecentlyCommand.slice(0, this.config.maxChatEditHistory);
    return recentlyCommand.map<InlineEditCommand>((commandStr) => {
      try {
        const command = JSON.parse(commandStr);
        if (typeof command === "object" && command.command) {
          return {
            command: command.command,
            context: command.context,
          };
        }
        return {
          command: commandStr,
        };
      } catch (error) {
        return {
          command: commandStr,
        };
      }
    });
  }

  private async addCommandHistory(userCommand: InlineEditCommand) {
    const commandStr = JSON.stringify(userCommand);
    const recentlyCommand = this.config.chatEditRecentlyCommand;
    const updatedRecentlyCommand = [commandStr]
      .concat(recentlyCommand.filter((item) => item !== commandStr))
      .slice(0, this.config.maxChatEditHistory);
    await this.config.updateChatEditRecentlyCommand(updatedRecentlyCommand);
  }

  private async deleteCommandHistory(command: string) {
    const recentlyCommand = this.getCommandHistory();
    const index = recentlyCommand.findIndex((item) => item.command === command);
    if (index !== -1) {
      recentlyCommand.splice(index, 1);
      await this.config.updateChatEditRecentlyCommand(recentlyCommand.map((command) => JSON.stringify(command)));
      this.updateQuickPickList();
    }
  }

  private async acceptCommand(command: string | undefined) {
    if (!command) {
      this.resultDeffer.resolve(undefined);
      return;
    }
    if (command && command.length > 200) {
      window.showErrorMessage("Command is too long.");
      this.resultDeffer.resolve(undefined);
      return;
    }

    const mentions = Array.from(new Set(parseUserCommand(command).mentions));

    const userCommand = {
      command,
      context: mentions
        .map<ChatEditFileContext | undefined>((item) => {
          if (this.fileContextLabelToUriMap.has(item)) {
            return {
              uri: this.fileContextLabelToUriMap.get(item) as string,
              referrer: item,
            };
          }
          return;
        })
        .filter((item): item is ChatEditFileContext => item !== undefined),
    };

    await this.addCommandHistory(userCommand);

    this.resultDeffer.resolve(userCommand);
    this.quickPick.hide();
  }

  private handleHidden() {
    this.fetchingSuggestedCommandCancellationTokenSource.cancel();
    if (this.filePick === undefined) {
      this.resultDeffer.resolve(undefined);
    }
  }

  private provideEditCommands() {
    this.client.chat.provideEditCommands(
      { location: this.editLocation },
      { commands: this.suggestedCommand, callback: () => this.updateQuickPickList() },
      this.fetchingSuggestedCommandCancellationTokenSource.token,
    );
  }

  private async handleTriggerItemButton(event: QuickPickItemButtonEvent<CommandQuickPickItem>) {
    const item = event.item;
    const button = event.button;
    if (button.iconPath instanceof ThemeIcon && button.iconPath.id === "settings-remove") {
      this.deleteCommandHistory(item.value);
    }

    if (button.iconPath instanceof ThemeIcon && button.iconPath.id === "edit") {
      this.updateQuickPickValue(item.value);
    }
  }
}

interface FileSelectionQuickPickItem extends QuickPickItem {
  uri: string;
}

interface FileSelectionResult {
  uri: string;
  label: string;
}

export class FileSelectionQuickPick {
  quickPick = window.createQuickPick<FileSelectionQuickPickItem>();
  private maxSearchFileResult = 30;
  private resultDeffer = new Deferred<FileSelectionResult | undefined>();

  constructor(private editor: TextEditor) {}

  private get workspaceFolder() {
    return workspace.getWorkspaceFolder(this.editor.document.uri);
  }

  start() {
    this.quickPick.title = "Enter file name to search";
    this.quickPick.buttons = [QuickInputButtons.Back];
    this.quickPick.ignoreFocusOut = true;
    // Quick pick items are always sorted by label. issue: https://github.com/microsoft/vscode/issues/73904
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (this.quickPick as any).sortByLabel = false;
    this.quickPick.onDidChangeValue((e) => this.updateFileList(e));
    this.quickPick.onDidAccept(() => this.handleAccept());
    this.quickPick.onDidHide(() => this.handleHidden());
    this.quickPick.onDidTriggerButton((e) => this.handleTriggerButton(e));
    this.quickPick.show();
    this.updateFileList("");
    return this.resultDeffer.promise;
  }

  private async updateFileList(val: string) {
    this.quickPick.busy = true;
    const fileList = await this.fetchFileList(val);
    this.quickPick.items = fileList;
    this.quickPick.busy = false;
  }

  private handleAccept() {
    const selection = this.quickPick.selectedItems[0];
    this.resultDeffer.resolve(selection ? { label: selection.label, uri: selection.uri } : undefined);
  }

  private handleHidden() {
    this.resultDeffer.resolve(undefined);
  }

  private handleTriggerButton(e: QuickInputButton) {
    if (e === QuickInputButtons.Back) {
      this.quickPick.hide();
    }
  }

  private async searchFileList(query: string) {
    if (!this.workspaceFolder) {
      return [];
    }
    const globPattern = caseInsensitivePattern(query);
    const fileList = await this.findFiles(globPattern, { maxResults: this.maxSearchFileResult });
    return fileList;
  }

  private findFiles = wrapCancelableFunction(
    findFiles,
    (args) => args[1]?.token,
    (args, token) => [args[0], { ...args[1], token }] as Parameters<typeof findFiles>,
  );

  private addFilesToList(files: Uri[], list: FileSelectionQuickPickItem[]) {
    files.forEach((item) => {
      const label = workspace.asRelativePath(item);
      const file = list.find((i) => i.label === label);
      if (file === undefined) {
        list.push({
          label: label,
          uri: item.toString(),
        });
      }
    });
    return list;
  }

  private getOpenEditor() {
    const list: FileSelectionQuickPickItem[] = [];

    if (this.editor) {
      const path = workspace.asRelativePath(this.editor.document.uri);
      list.push({
        label: path,
        uri: this.editor.document.uri.toString(),
      });
    }

    return this.addFilesToList(getOpenTabsUri(), list);
  }

  private async fetchFileList(query: string) {
    const list: FileSelectionQuickPickItem[] = [];

    list.push(...this.getOpenEditor());

    if (list.length > 0) {
      list.push({
        label: "",
        uri: "",
        kind: QuickPickItemKind.Separator,
        alwaysShow: true,
      });
    }
    const fileList = await this.searchFileList(query);
    return this.addFilesToList(fileList, list);
  }
}

const getOpenTabsUri = (): Uri[] => {
  return window.tabGroups.all
    .flatMap((group) => group.tabs.map((tab) => (tab.input instanceof TabInputText ? tab.input.uri : null)))
    .filter((item): item is Uri => item !== null);
};
