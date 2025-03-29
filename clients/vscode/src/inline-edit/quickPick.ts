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
  commands,
  SymbolInformation,
  SymbolKind,
  DocumentSymbol,
} from "vscode";
import { Config } from "../Config";
import { ChatEditCommand, ChatEditFileContext } from "tabby-agent";
import { Deferred, InlineEditParseResult, parseUserCommand, replaceLastOccurrence } from "./util";
import { Client } from "../lsp/client";
import { Location } from "vscode-languageclient";
import { caseInsensitivePattern, findFiles } from "../findFiles";

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
    private editor: TextEditor,
    private editLocation: Location,
  ) {
    this.editLocation = editLocation;
  }

  start() {
    // Create a new deferred object for each start call
    this.resultDeffer = new Deferred<InlineEditCommand | undefined>();

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
      // User typed "@file", open the file picker directly
      await this.openFilePick();
      return;
    } else if (mentionText.toLowerCase() === "symbol") {
      // User typed "@symbol", open the symbol picker directly
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

    // Show loading indicator while fetching initial items
    contextPicker.busy = true;

    // Add default file list (open tabs and workspace files)
    const fileItems = await UserCommandQuickpick.getFileItems("");

    // Set initial items
    contextPicker.items = [...contextTypeItems, ...fileItems];
    contextPicker.busy = false;

    // Allow filtering
    contextPicker.onDidChangeValue(async (value) => {
      if (value) {
        // Show loading indicator
        contextPicker.busy = true;

        // Get filtered file items
        const filteredFileItems = await UserCommandQuickpick.getFileItems(value);

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

  /**
   * Helper method to get file items with consistent formatting
   * This is used by both context picker and file selection picker
   */
  static async getFileItems(query: string, maxResults = 20): Promise<(QuickPickItem & { uri?: string })[]> {
    // Get open tabs first
    const openTabs = getOpenTabsUri();
    const openTabsUriStrings = openTabs.map((uri) => uri.toString());

    // Create items for open tabs
    const openTabItems = openTabs.map((uri) => ({
      label: `$(file) ${workspace.asRelativePath(uri)}`,
      description: "Open in editor",
      buttons: [{ iconPath: new ThemeIcon("edit") }],
      uri: uri.toString(),
    }));

    // Always search for files, even when there's no query
    // Use a default pattern when no query is provided
    const globPattern = query ? caseInsensitivePattern(query) : "**/*";
    const fileList = await findFiles(globPattern, { maxResults });

    // Create items for search results
    const fileItems = fileList.map((uri) => {
      const uriString = uri.toString();
      const isOpenInEditor = openTabsUriStrings.includes(uriString);

      return {
        label: `$(file) ${workspace.asRelativePath(uri)}`,
        description: isOpenInEditor ? "Open in editor" : undefined,
        buttons: isOpenInEditor ? [{ iconPath: new ThemeIcon("edit") }] : undefined,
        uri: uriString,
      };
    });

    // Filter out duplicates
    const seen = new Set(openTabsUriStrings);
    const uniqueFileItems = fileItems.filter((item) => {
      if (!item.uri || seen.has(item.uri)) {
        return false;
      }
      seen.add(item.uri);
      return true;
    });

    return [...openTabItems, ...uniqueFileItems];
  }

  private async openFilePick() {
    this.filePick = new FileSelectionQuickPick();
    const file = await this.filePick.start();
    this.quickPick.show();
    if (file) {
      this.updateQuickPickValue(this.quickPick.value + `${file.label} `);
      this.fileContextLabelToUriMap.set(file.label, { uri: file.uri });
      // Update the quick pick list to show the command with the file name
      this.updateQuickPickList();
    } else {
      // remove `@` when user select no file
      this.updateQuickPickValue(replaceLastOccurrence(this.quickPick.value, "@", ""));
    }
    this.filePick = undefined;
  }

  private async openSymbolPick() {
    this.symbolPick = new SymbolSelectionQuickPick(this.editor);
    const symbol = await this.symbolPick.start();
    this.quickPick.show();
    if (symbol) {
      this.updateQuickPickValue(this.quickPick.value + `${symbol.label} `);
      this.fileContextLabelToUriMap.set(symbol.label, { uri: symbol.uri, range: symbol.range });
      // Update the quick pick list to show the command with the symbol name
      this.updateQuickPickList();
    } else {
      // remove `@` when user select no symbol
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

    // Check if we're in the middle of a file or symbol selection
    const inFileOrSymbolSelection = this.filePick !== undefined || this.symbolPick !== undefined;

    // Check if a file has been directly selected
    const fileDirectlySelected = this.directFileSelected;

    // Check if we're about to show the context picker
    const aboutToShowContextPicker = this.showingContextPicker;

    // Only resolve with undefined if we're not in the middle of a file or symbol selection,
    // a file hasn't been directly selected, and we're not about to show the context picker
    if (!inFileOrSymbolSelection && !fileDirectlySelected && !aboutToShowContextPicker) {
      this.resultDeferred.resolve(undefined);
    }
    // Otherwise, don't resolve here, let the accept handler resolve with the command

    // Reset the flag
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
    const fileList = await this.fetchFileList(val);
    this.quickPick.items = fileList;
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

  private async fetchFileList(query: string): Promise<FileSelectionQuickPickItem[]> {
    // Use the shared static getFileItems method from UserCommandQuickpick
    const fileItems = await UserCommandQuickpick.getFileItems(query, this.maxSearchFileResult);

    // Convert to FileSelectionQuickPickItem
    return fileItems.map((item: QuickPickItem & { uri?: string }) => ({
      label: item.label,
      description: item.description,
      buttons: item.buttons,
      uri: item.uri || "",
      kind: item.kind,
      alwaysShow: item.alwaysShow,
    })) as FileSelectionQuickPickItem[];
  }
}

const getOpenTabsUri = (): Uri[] => {
  return window.tabGroups.all
    .flatMap((group) => group.tabs.map((tab) => (tab.input instanceof TabInputText ? tab.input.uri : null)))
    .filter((item): item is Uri => item !== null);
};

interface SymbolSelectionQuickPickItem extends QuickPickItem {
  uri: string;
  range?: { start: { line: number; character: number }; end: { line: number; character: number } };
}

interface SymbolSelectionResult {
  uri: string;
  label: string;
  range?: { start: { line: number; character: number }; end: { line: number; character: number } };
}

export class SymbolSelectionQuickPick {
  quickPick = window.createQuickPick<SymbolSelectionQuickPickItem>();
  private resultDeferred = new Deferred<SymbolSelectionResult | undefined>();

  constructor(private editor: TextEditor) {}

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

  private async fetchSymbolList(query: string): Promise<SymbolSelectionQuickPickItem[]> {
    try {
      // Get document symbols from the current file
      const rawDocumentSymbols =
        (await commands.executeCommand<DocumentSymbol[] | SymbolInformation[]>(
          "vscode.executeDocumentSymbolProvider",
          this.editor.document.uri,
        )) || [];

      // Convert DocumentSymbol[] to SymbolInformation[] if needed
      const documentSymbols = this.convertToSymbolInformation(rawDocumentSymbols);

      // Get workspace symbols if query is provided
      const workspaceSymbols = query
        ? (await commands.executeCommand<SymbolInformation[]>("vscode.executeWorkspaceSymbolProvider", query)) || []
        : [];

      // Combine and filter symbols
      const allSymbols = [...documentSymbols, ...workspaceSymbols];
      const filteredSymbols = this.filterSymbols(allSymbols, query);

      // Convert to QuickPickItems with appropriate icons
      return filteredSymbols.map((symbol) => ({
        label: symbol.name,
        description: this.getSymbolDescription(symbol),
        iconPath: this.getSymbolIcon(symbol.kind),
        uri: symbol.location.uri.toString(),
        range: {
          start: {
            line: symbol.location.range.start.line,
            character: symbol.location.range.start.character,
          },
          end: {
            line: symbol.location.range.end.line,
            character: symbol.location.range.end.character,
          },
        },
      }));
    } catch (error) {
      // Error handling silently
      return [];
    }
  }

  private convertToSymbolInformation(symbols: (DocumentSymbol | SymbolInformation)[]): SymbolInformation[] {
    const result: SymbolInformation[] = [];

    // Check if we have DocumentSymbol[] or SymbolInformation[]
    if (symbols.length > 0 && symbols[0] && "children" in symbols[0]) {
      // We have DocumentSymbol[], need to flatten
      this.flattenDocumentSymbols(symbols as DocumentSymbol[], "", this.editor.document.uri, result);
    } else if (symbols.length > 0) {
      // We already have SymbolInformation[]
      result.push(...(symbols as SymbolInformation[]));
    }

    return result;
  }

  private flattenDocumentSymbols(
    symbols: DocumentSymbol[],
    containerName: string,
    uri: Uri,
    result: SymbolInformation[],
  ): void {
    for (const symbol of symbols) {
      const fullName = containerName ? `${containerName}.${symbol.name}` : symbol.name;

      // Create a SymbolInformation from DocumentSymbol
      result.push({
        name: symbol.name,
        kind: symbol.kind,
        containerName: containerName,
        location: {
          uri: uri,
          range: symbol.range,
        },
      } as SymbolInformation);

      // Process children recursively
      if (symbol.children && symbol.children.length > 0) {
        this.flattenDocumentSymbols(symbol.children, fullName, uri, result);
      }
    }
  }

  private getSymbolDescription(symbol: SymbolInformation): string {
    const containerName = symbol.containerName ? `${symbol.containerName}` : "";
    const fileName = workspace.asRelativePath(symbol.location.uri);

    return containerName ? `${containerName} (${fileName})` : fileName;
  }

  private getSymbolIcon(kind: SymbolKind): ThemeIcon {
    // Map symbol kinds to appropriate theme icons
    switch (kind) {
      case SymbolKind.File:
        return new ThemeIcon("file");
      case SymbolKind.Module:
        return new ThemeIcon("package");
      case SymbolKind.Namespace:
        return new ThemeIcon("symbol-namespace");
      case SymbolKind.Class:
        return new ThemeIcon("symbol-class");
      case SymbolKind.Method:
        return new ThemeIcon("symbol-method");
      case SymbolKind.Property:
        return new ThemeIcon("symbol-property");
      case SymbolKind.Field:
        return new ThemeIcon("symbol-field");
      case SymbolKind.Constructor:
        return new ThemeIcon("symbol-constructor");
      case SymbolKind.Enum:
        return new ThemeIcon("symbol-enum");
      case SymbolKind.Interface:
        return new ThemeIcon("symbol-interface");
      case SymbolKind.Function:
        return new ThemeIcon("symbol-method");
      case SymbolKind.Variable:
        return new ThemeIcon("symbol-variable");
      case SymbolKind.Constant:
        return new ThemeIcon("symbol-constant");
      case SymbolKind.String:
        return new ThemeIcon("symbol-string");
      case SymbolKind.Number:
        return new ThemeIcon("symbol-number");
      case SymbolKind.Boolean:
        return new ThemeIcon("symbol-boolean");
      case SymbolKind.Array:
        return new ThemeIcon("symbol-array");
      case SymbolKind.Object:
        return new ThemeIcon("symbol-object");
      case SymbolKind.Key:
        return new ThemeIcon("symbol-key");
      case SymbolKind.Null:
        return new ThemeIcon("symbol-null");
      case SymbolKind.EnumMember:
        return new ThemeIcon("symbol-enum-member");
      case SymbolKind.Struct:
        return new ThemeIcon("symbol-struct");
      case SymbolKind.Event:
        return new ThemeIcon("symbol-event");
      case SymbolKind.Operator:
        return new ThemeIcon("symbol-operator");
      case SymbolKind.TypeParameter:
        return new ThemeIcon("symbol-parameter");
      default:
        return new ThemeIcon("symbol-misc");
    }
  }

  private filterSymbols(symbols: SymbolInformation[], query: string): SymbolInformation[] {
    // Remove duplicates
    const uniqueSymbols = this.removeDuplicateSymbols(symbols);

    if (!query) {
      return uniqueSymbols.slice(0, 50); // Limit to 20 symbols when no query
    }

    const lowerQuery = query.toLowerCase();
    const filtered = uniqueSymbols.filter((s) => s.name.toLowerCase().includes(lowerQuery));

    // Sort by match quality
    return filtered
      .sort((a, b) => {
        const aName = a.name.toLowerCase();
        const bName = b.name.toLowerCase();

        // Exact match
        if (aName === lowerQuery && bName !== lowerQuery) return -1;
        if (bName === lowerQuery && aName !== lowerQuery) return 1;

        // Starts with query
        if (aName.startsWith(lowerQuery) && !bName.startsWith(lowerQuery)) return -1;
        if (bName.startsWith(lowerQuery) && !aName.startsWith(lowerQuery)) return 1;

        // Current file symbols first
        const aIsCurrentFile = a.location.uri.toString() === this.editor.document.uri.toString();
        const bIsCurrentFile = b.location.uri.toString() === this.editor.document.uri.toString();
        if (aIsCurrentFile && !bIsCurrentFile) return -1;
        if (bIsCurrentFile && !aIsCurrentFile) return 1;

        // Shorter name is better
        return a.name.length - b.name.length;
      })
      .slice(0, 50); // Limit to 20 results
  }

  private removeDuplicateSymbols(symbols: SymbolInformation[]): SymbolInformation[] {
    const seen = new Set<string>();
    return symbols.filter((symbol) => {
      const key = `${symbol.name}-${symbol.containerName}-${symbol.location.uri.toString()}`;
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }
}
