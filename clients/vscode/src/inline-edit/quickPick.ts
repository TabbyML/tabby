import {
  CancellationToken,
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
import { listSymbols } from "../findSymbols";
import { Config } from "../Config";
import { ChatEditCommand, ChatEditFileContext } from "tabby-agent";
import { Deferred, InlineEditParseResult, parseUserCommand, replaceLastOccurrence } from "./util";
import { Client } from "../lsp/client";
import { Location } from "vscode-languageclient";
import { listFiles } from "../findFiles";
import { wrapCancelableFunction } from "../cancelableFunction";

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

const getFileItems = async (query: string, maxResults = 20): Promise<FileSelectionQuickPickItem[]> => {
  const fileList = await wrappedListFiles(query, maxResults);
  const fileItems = fileList.map((fileItem) => {
    const uriString = fileItem.uri.toString();
    return {
      label: `$(file) ${workspace.asRelativePath(fileItem.uri)}`,
      description: fileItem.isOpenedInEditor ? "Open in editor" : undefined,
      buttons: fileItem.isOpenedInEditor ? [{ iconPath: new ThemeIcon("edit") }] : undefined,
      uri: uriString,
    };
  });
  return fileItems;
};

export class UserCommandQuickpick {
  quickPick = window.createQuickPick<CommandQuickPickItem>();
  private suggestedCommand: ChatEditCommand[] = [];
  private resultDeferred = new Deferred<InlineEditCommand | undefined>();
  private fetchingSuggestedCommandCancellationTokenSource = new CancellationTokenSource();
  private lastInputValue = "";
  private filePick: FileSelectionQuickPick | undefined;
  private symbolPick: SymbolSelectionQuickPick | undefined;
  private fileContextLabelToUriMap = new Map<string, Omit<ChatEditFileContext, "referrer">>();
  private directFileSelected = false; // Flag to indicate a file has been selected directly
  private showingContextPicker = false; // Flag to indicate we're about to show the context picker

  // Properties to store pending changes
  private _pendingValue: string | undefined;
  private _pendingFileContext: ({ label: string } & Omit<ChatEditFileContext, "referrer">) | undefined;

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

    // Return a promise that will be resolved when the user accepts a command
    return this.resultDeferred.promise;
  }

  private get inputParseResult(): InlineEditParseResult {
    return parseUserCommand(this.quickPick.value);
  }

  private handleValueChange() {
    const { mentionQuery } = this.inputParseResult;
    if (mentionQuery === "") {
      // Set the flag to indicate we're about to show the context picker
      this.showingContextPicker = true;

      // Instead of immediately showing the context picker, just hide the current quick pick
      // This will prevent the handleHidden method from resolving the promise with undefined
      this.quickPick.hide();

      // Then show the context picker in a separate async operation
      setTimeout(() => {
        this.showContextPicker();
      }, 0);
    } else {
      this.updateQuickPickList();
      this.updateQuickPickValue(this.quickPick.value);
    }
  }

  private async showContextPicker() {
    // Get the current command value
    const currentValue = this.quickPick.value;

    // First check if we should show the file picker or symbol picker directly
    const mentionText = currentValue.substring(currentValue.lastIndexOf("@") + 1).trim();

    if (mentionText.toLowerCase() === "file") {
      await this.openFilePick();
      return;
    } else if (mentionText.toLowerCase() === "symbol") {
      await this.openSymbolPick();
      return;
    }

    // Otherwise show the combined picker
    // Create a combined quick pick that shows both context types and files
    const contextPicker = window.createQuickPick<QuickPickItem & { type?: string; uri?: string }>();
    contextPicker.title = "Select context or file";

    // Add context type options with separators
    const contextTypeItems: (QuickPickItem & { type?: string })[] = [
      { label: "$(folder) File", description: "Reference a file in the workspace", type: "file" },
      { label: "$(symbol-class) Symbol", description: "Reference a symbol in the current file", type: "symbol" },
      { label: "", kind: QuickPickItemKind.Separator },
    ];

    contextPicker.busy = true;
    const fileItems = await getFileItems("", 20);
    contextPicker.items = [...contextTypeItems, ...fileItems];
    contextPicker.busy = false;
    contextPicker.onDidChangeValue(async (value) => {
      if (value) {
        contextPicker.busy = true;

        const filteredFileItems = await getFileItems(value, 20);

        // Update items
        contextPicker.items = [...contextTypeItems, ...filteredFileItems];
        contextPicker.busy = false;
      } else {
        // Reset to default items
        contextPicker.items = [...contextTypeItems, ...fileItems];
      }
    });

    const deferred = new Deferred<{ type?: "file" | "symbol"; uri?: string; label?: string } | undefined>();

    contextPicker.onDidAccept(() => {
      const selected = contextPicker.selectedItems[0];

      if (selected?.type === "file") {
        // User selected "File" option
        deferred.resolve({ type: "file" });
      } else if (selected?.type === "symbol") {
        // User selected "Symbol" option
        deferred.resolve({ type: "symbol" });
      } else if (selected?.uri) {
        // User selected a file directly
        const uri = selected.uri;
        const label = selected.label.replace(/^\$\(file\) /, ""); // Remove the file icon prefix

        // Add the file to the input and context map
        const newValue = currentValue + `${label} `;

        // Store the new value and file context for later use
        this._pendingValue = newValue;
        this._pendingFileContext = { label, uri };

        // Set the flag to indicate a file has been selected directly
        this.directFileSelected = true;

        // Return undefined to indicate we've handled the selection directly
        deferred.resolve(undefined);
      } else {
        deferred.resolve(undefined);
      }

      contextPicker.hide();
    });

    contextPicker.onDidHide(() => {
      // Always resolve on hide to ensure we don't leave hanging promises
      deferred.resolve(undefined);
      contextPicker.dispose();
    });

    contextPicker.show();

    const result = await deferred.promise;

    // Show the main quick pick again
    this.quickPick.show();

    // Apply any pending changes
    if (this._pendingValue) {
      this.updateQuickPickValue(this._pendingValue);
      if (this._pendingFileContext) {
        this.fileContextLabelToUriMap.set(this._pendingFileContext.label, {
          uri: this._pendingFileContext.uri,
          range: this._pendingFileContext.range,
        });
      }
      this._pendingValue = undefined;
      this._pendingFileContext = undefined;

      // Update the quick pick list to show the command with the file name
      this.updateQuickPickList();
    }

    if (result?.type === "file") {
      await this.openFilePick();
    } else if (result?.type === "symbol") {
      await this.openSymbolPick();
    } else {
      // User cancelled or selected a file directly (already handled)
      // If the @ is still there, remove it
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
      this.updateQuickPickValue(this.quickPick.value + `${file.label} `);
      this.fileContextLabelToUriMap.set(file.label, { uri: file.uri });
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
      this.updateQuickPickValue(this.quickPick.value + `${symbol.label} `);
      this.fileContextLabelToUriMap.set(symbol.label, { uri: symbol.uri, range: symbol.range });
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
            this.fileContextLabelToUriMap.set(context.referrer, {
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
    // Get the command from the selected item or use the current value if no item is selected
    const command = this.quickPick.selectedItems[0]?.value || this.quickPick.value;

    // Reset the direct file selection flag
    this.directFileSelected = false;

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
          if (this.fileContextLabelToUriMap.has(item)) {
            const contextInfo = this.fileContextLabelToUriMap.get(item);
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
    const fileDirectlySelected = this.directFileSelected;
    const aboutToShowContextPicker = this.showingContextPicker;

    if (!inFileOrSymbolSelection && !fileDirectlySelected && !aboutToShowContextPicker) {
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
}

interface FileSelectionResult {
  uri: string;
  label: string;
}

export class FileSelectionQuickPick {
  quickPick = window.createQuickPick<FileSelectionQuickPickItem>();
  private maxSearchFileResult = 30;
  private resultDeferred = new Deferred<FileSelectionResult | undefined>();

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
    this.quickPick.items = await getFileItems(val);
    this.quickPick.busy = false;
  }

  private handleAccept() {
    const selection = this.quickPick.selectedItems[0];
    if (selection) {
      // Remove the file icon prefix from the label, consistent with UserCommandQuickpick
      const label = selection.label.replace(/^\$\(file\) /, "");
      this.resultDeferred.resolve({ label, uri: selection.uri });
    } else {
      this.resultDeferred.resolve(undefined);
    }
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
  range?: Range;
}

interface SymbolSelectionResult {
  uri: string;
  label: string;
  range?: Range;
}

export class SymbolSelectionQuickPick {
  quickPick = window.createQuickPick<SymbolSelectionQuickPickItem>();
  private resultDeferred = new Deferred<SymbolSelectionResult | undefined>();

  start() {
    this.quickPick.title = "Enter symbol name to search";
    this.quickPick.placeholder = "Type to filter symbols in the current file";
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
    const selection = this.quickPick.selectedItems[0];
    this.resultDeferred.resolve(
      selection
        ? {
            label: selection.label,
            uri: selection.uri,
            range: selection.range,
          }
        : undefined,
    );
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
            description: symbol.containerName || "",
            iconPath: symbol.kindIcon,
            uri: symbol.location.uri.toString(),
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
