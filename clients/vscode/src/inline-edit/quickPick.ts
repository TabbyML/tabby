import {
  CancellationTokenSource,
  QuickInputButton,
  QuickInputButtons,
  QuickPickItem,
  QuickPickItemButtonEvent,
  QuickPickItemKind,
  Range,
  ThemeIcon,
  window,
  workspace,
} from "vscode";
import path from "path";
import { ChatEditCommand, ChatEditFileContext } from "tabby-agent";
import { listSymbols } from "../findSymbols";
import { Config } from "../Config";
import { Deferred } from "../deferred";
import { Client } from "../lsp/client";
import { Location } from "vscode-languageclient";
import { listFiles } from "../findFiles";
import { wrapCancelableFunction } from "../cancelableFunction";
import { InlineEditParseResult, parseUserCommand, replaceLastOccurrence } from "./util";

export interface InlineEditCommand {
  command: string;
  context?: ChatEditFileContext[];
}

interface CommandQuickPickItem extends QuickPickItem {
  value: string;
}

/**
 * Helper method to get file items with consistent formatting
 * This is used by both context picker and file selection picker
 */
const wrappedListFiles = wrapCancelableFunction(
  listFiles,
  (args) => args[2],
  (args, token) => {
    args[2] = token;
    return args;
  },
);

const getFileItems = async (query: string, maxResults: number): Promise<FileSelectionQuickPickItem[]> => {
  const fileList = await wrappedListFiles(query, maxResults);
  const fileItems: FileSelectionQuickPickItem[] = fileList.map((fileItem) => {
    const relativePath = workspace.asRelativePath(fileItem.uri);
    const basename = path.basename(fileItem.uri.fsPath);
    const dirname = path.dirname(relativePath);
    return {
      label: `$(file) ${basename}`,
      description: dirname === "." ? "" : dirname,
      alwaysShow: true,
      referer: relativePath,
      uri: fileItem.uri.toString(),
      isOpenedInEditor: fileItem.isOpenedInEditor,
    };
  });
  const activeFilesIndex = fileItems.findIndex((item) => item.isOpenedInEditor);
  if (activeFilesIndex != -1) {
    fileItems.splice(activeFilesIndex, 0, {
      label: `active files`,
      kind: QuickPickItemKind.Separator,
      referer: "",
      uri: "",
      isOpenedInEditor: true,
    });
  }
  const searchResultsIndex = fileItems.findIndex((item) => !item.isOpenedInEditor);
  if (searchResultsIndex != -1) {
    fileItems.splice(searchResultsIndex, 0, {
      label: `search results`,
      kind: QuickPickItemKind.Separator,
      referer: "",
      uri: "",
      isOpenedInEditor: false,
    });
  }
  return fileItems;
};

interface ContextQuickPickItem extends QuickPickItem {
  type: undefined | "file" | "symbol";
}

export class UserCommandQuickpick {
  quickPick = window.createQuickPick<CommandQuickPickItem>();
  private suggestedCommand: ChatEditCommand[] = [];
  private resultDeferred = new Deferred<InlineEditCommand | undefined>();
  private fetchingSuggestedCommandCancellationTokenSource = new CancellationTokenSource();
  private lastInputValue = "";
  private filePick: FileSelectionQuickPick | undefined;
  private symbolPick: SymbolSelectionQuickPick | undefined;
  private showingContextPicker = false;
  private referenceMap = new Map<string, Omit<ChatEditFileContext, "referrer">>();

  constructor(
    private client: Client,
    private config: Config,
    private editLocation: Location,
  ) {}

  start() {
    this.quickPick.title = "Enter the command for editing (type @ to include file or symbol)";
    this.quickPick.matchOnDescription = true;
    this.quickPick.onDidChangeValue(() => this.handleValueChange());
    this.quickPick.onDidAccept(() => this.handleAccept());
    this.quickPick.onDidHide(() => this.handleHidden());
    this.quickPick.onDidTriggerItemButton((e) => this.handleTriggerItemButton(e));

    this.quickPick.show();
    this.quickPick.ignoreFocusOut = true;
    this.provideEditCommands();
    return this.resultDeferred.promise;
  }

  private get inputParseResult(): InlineEditParseResult {
    return parseUserCommand(this.quickPick.value);
  }

  private handleValueChange() {
    const { mentionQuery } = this.inputParseResult;
    if (mentionQuery === "") {
      this.showingContextPicker = true;
      this.quickPick.hide();
      this.showContextPicker();
    } else {
      this.updateQuickPickList();
      this.updateQuickPickValue(this.quickPick.value);
    }
  }

  private async showContextPicker() {
    const contextPicker = window.createQuickPick<ContextQuickPickItem | FileSelectionQuickPickItem>();
    contextPicker.title = "Select context or file";
    contextPicker.buttons = [QuickInputButtons.Back];
    contextPicker.ignoreFocusOut = true;
    // Quick pick items are always sorted by label. issue: https://github.com/microsoft/vscode/issues/73904
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (contextPicker as any).sortByLabel = false;

    const contextTypeItems: ContextQuickPickItem[] = [
      { label: "context", kind: QuickPickItemKind.Separator, type: undefined },
      { label: "$(folder) Files $(chevron-right)", type: "file" },
      { label: "$(symbol-class) Symbols $(chevron-right)", type: "symbol" },
    ];

    contextPicker.busy = true;
    const fileItemsMaxResult = 20;
    const fileItems = await getFileItems("", fileItemsMaxResult);
    contextPicker.items = [...contextTypeItems, ...fileItems];
    contextPicker.busy = false;
    contextPicker.onDidChangeValue(async (value) => {
      if (value) {
        contextPicker.busy = true;
        const filteredFileItems = await getFileItems(value, fileItemsMaxResult);
        contextPicker.items = [...contextTypeItems, ...filteredFileItems];
        contextPicker.busy = false;
      } else {
        contextPicker.items = [...contextTypeItems, ...fileItems];
      }
    });

    const deferred = new Deferred<ContextQuickPickItem | FileSelectionQuickPickItem | undefined>();

    contextPicker.onDidAccept(() => {
      deferred.resolve(contextPicker.selectedItems[0]);
      contextPicker.hide();
    });

    contextPicker.onDidHide(() => {
      deferred.resolve(undefined);
      contextPicker.dispose();
    });

    contextPicker.onDidTriggerButton((e: QuickInputButton) => {
      if (e === QuickInputButtons.Back) {
        contextPicker.hide();
      }
    });

    contextPicker.show();

    const result = await deferred.promise;

    if (result && "type" in result) {
      if (result.type === "file") {
        await this.openFilePick();
      } else if (result.type === "symbol") {
        await this.openSymbolPick();
      }
    } else if (result && "uri" in result) {
      this.quickPick.show();
      this.updateQuickPickValue(this.quickPick.value + `${result.referer} `);
      this.referenceMap.set(result.referer, { uri: result.uri });
    } else {
      this.quickPick.show();
      if (this.quickPick.value.endsWith("@")) {
        this.updateQuickPickValue(replaceLastOccurrence(this.quickPick.value, "@", ""));
      }
    }
  }

  private async openFilePick() {
    this.filePick = new FileSelectionQuickPick();
    const file = await this.filePick.start();
    this.quickPick.show();
    if (file) {
      this.updateQuickPickValue(this.quickPick.value + `${file.referer} `);
      this.referenceMap.set(file.referer, { uri: file.uri });
    } else {
      this.updateQuickPickValue(replaceLastOccurrence(this.quickPick.value, "@", ""));
    }
    this.filePick = undefined;
  }

  private async openSymbolPick() {
    this.symbolPick = new SymbolSelectionQuickPick();
    const symbol = await this.symbolPick.start();
    this.quickPick.show();
    if (symbol) {
      this.updateQuickPickValue(this.quickPick.value + `${symbol.referer} `);
      this.referenceMap.set(symbol.referer, { uri: symbol.uri, range: symbol.range });
    } else {
      this.updateQuickPickValue(replaceLastOccurrence(this.quickPick.value, "@", ""));
    }
    this.symbolPick = undefined;
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
    list.push({
      label: "commands",
      value: "",
      kind: QuickPickItemKind.Separator,
    });
    list.push(
      ...this.suggestedCommand.map((item) => ({
        label: item.label,
        value: item.command,
        iconPath: new ThemeIcon("sparkle"),
        description: item.source === "preset" ? item.command : "",
      })),
    );
    list.push({
      label: "history",
      value: "",
      kind: QuickPickItemKind.Separator,
    });
    const recentlyCommandToAdd = this.getCommandHistory().filter((item) => !list.find((i) => i.value === item.command));
    recentlyCommandToAdd.forEach((command) => {
      if (command.context) {
        command.context.forEach((context) => {
          if (!this.referenceMap.has(context.referrer)) {
            // this context maybe outdated
            this.referenceMap.set(context.referrer, {
              uri: context.uri,
              range: context.range,
            });
          }
        });
      }
    });
    list.push(
      ...recentlyCommandToAdd.map((item) => ({
        label: item.command,
        value: item.command,
        iconPath: new ThemeIcon("history"),
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
    const command = this.quickPick.selectedItems[0]?.value || this.quickPick.value;
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
      this.resultDeferred.resolve(undefined);
      return;
    }
    if (command && command.length > 200) {
      window.showErrorMessage("Command is too long.");
      this.resultDeferred.resolve(undefined);
      return;
    }

    const parseResult = parseUserCommand(command);
    const mentionTexts = parseResult.mentions?.map((mention) => mention.text) || [];
    const uniqueMentionTexts = Array.from(new Set(mentionTexts));

    const userCommand = {
      command,
      context: uniqueMentionTexts
        .map<ChatEditFileContext | undefined>((item) => {
          if (this.referenceMap.has(item)) {
            const contextInfo = this.referenceMap.get(item);
            if (contextInfo) {
              return {
                uri: contextInfo.uri,
                referrer: item,
                range: contextInfo.range,
              };
            }
          }
          return;
        })
        .filter((item): item is ChatEditFileContext => item !== undefined),
    };

    await this.addCommandHistory(userCommand);

    this.resultDeferred.resolve(userCommand);
    this.quickPick.hide();
  }

  private handleHidden() {
    this.fetchingSuggestedCommandCancellationTokenSource.cancel();
    const inFileOrSymbolSelection = this.filePick !== undefined || this.symbolPick !== undefined;
    if (!inFileOrSymbolSelection && !this.showingContextPicker) {
      this.resultDeferred.resolve(undefined);
    }
    this.showingContextPicker = false;
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
  referer: string;
  isOpenedInEditor: boolean;
}

export class FileSelectionQuickPick {
  quickPick = window.createQuickPick<FileSelectionQuickPickItem>();
  private maxSearchFileResult = 50;
  private resultDeferred = new Deferred<FileSelectionQuickPickItem | undefined>();

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
    return this.resultDeferred.promise;
  }

  private async updateFileList(val: string) {
    this.quickPick.busy = true;
    this.quickPick.items = await getFileItems(val, this.maxSearchFileResult);
    this.quickPick.busy = false;
  }

  private handleAccept() {
    this.resultDeferred.resolve(this.quickPick.selectedItems[0]);
  }

  private handleHidden() {
    this.resultDeferred.resolve(undefined);
  }

  private handleTriggerButton(e: QuickInputButton) {
    if (e === QuickInputButtons.Back) {
      this.quickPick.hide();
    }
  }
}

interface SymbolSelectionQuickPickItem extends QuickPickItem {
  uri: string;
  referer: string;
  range?: Range;
}

export class SymbolSelectionQuickPick {
  quickPick = window.createQuickPick<SymbolSelectionQuickPickItem>();
  private resultDeferred = new Deferred<SymbolSelectionQuickPickItem | undefined>();

  start() {
    this.quickPick.title = "Enter symbol name to search";
    this.quickPick.buttons = [QuickInputButtons.Back];
    this.quickPick.ignoreFocusOut = true;
    this.quickPick.onDidChangeValue((e) => this.updateSymbolList(e));
    this.quickPick.onDidAccept(() => this.handleAccept());
    this.quickPick.onDidHide(() => this.handleHidden());
    this.quickPick.onDidTriggerButton((e) => this.handleTriggerButton(e));
    this.quickPick.show();
    this.updateSymbolList("");
    return this.resultDeferred.promise;
  }

  private async updateSymbolList(query: string) {
    this.quickPick.busy = true;
    const symbolList = await this.fetchSymbolList(query);
    this.quickPick.items = symbolList;
    this.quickPick.busy = false;
  }

  private handleAccept() {
    this.resultDeferred.resolve(this.quickPick.selectedItems[0]);
  }

  private handleHidden() {
    this.resultDeferred.resolve(undefined);
  }

  private handleTriggerButton(e: QuickInputButton) {
    if (e === QuickInputButtons.Back) {
      this.quickPick.hide();
    }
  }

  private listSymbols = wrapCancelableFunction(
    listSymbols,
    () => undefined,
    (args) => args,
  );

  private async fetchSymbolList(query: string): Promise<SymbolSelectionQuickPickItem[]> {
    if (!window.activeTextEditor) {
      return [];
    }
    try {
      const symbols = await this.listSymbols(window.activeTextEditor.document.uri, query, 50);
      return symbols.map(
        (symbol) =>
          ({
            label: symbol.name,
            description: symbol.containerName,
            iconPath: symbol.kindIcon,
            uri: symbol.location.uri.toString(),
            referer: symbol.name.replace(/\s/g, "_").replace(/@/g, ""),
            // FIXME(icycode): extract type conversion utils
            range: symbol.location.range
              ? {
                  start: {
                    line: symbol.location.range.start.line,
                    character: symbol.location.range.start.character,
                  },
                  end: {
                    line: symbol.location.range.end.line,
                    character: symbol.location.range.end.character,
                  },
                }
              : undefined,
          }) as SymbolSelectionQuickPickItem,
      );
    } catch (error) {
      return [];
    }
  }
}
