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
  private symbolPick: SymbolSelectionQuickPick | undefined;
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
    this.quickPick.title = "Enter the command for editing (type @ to include file or symbol)";
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
      this.showContextPicker();
    } else {
      this.updateQuickPickList();
      this.updateQuickPickValue(this.quickPick.value);
    }
  }

  private async showContextPicker() {
    // Show a quick pick to choose between file and symbol
    const contextPicker = window.createQuickPick<QuickPickItem>();
    contextPicker.title = "Select context type";
    contextPicker.items = [
      { label: "File", description: "Reference a file in the workspace" },
      { label: "Symbol", description: "Reference a symbol in the current file" },
    ];

    const deferred = new Deferred<"file" | "symbol" | undefined>();

    contextPicker.onDidAccept(() => {
      const selected = contextPicker.selectedItems[0]?.label;
      if (selected === "File") {
        deferred.resolve("file");
      } else if (selected === "Symbol") {
        deferred.resolve("symbol");
      } else {
        deferred.resolve(undefined);
      }
      contextPicker.hide();
    });

    contextPicker.onDidHide(() => {
      deferred.resolve(undefined);
      contextPicker.dispose();
    });

    contextPicker.show();

    const contextType = await deferred.promise;

    if (contextType === "file") {
      await this.openFilePick();
    } else if (contextType === "symbol") {
      await this.openSymbolPick();
    } else {
      // User cancelled, remove the @ character
      this.updateQuickPickValue(replaceLastOccurrence(this.quickPick.value, "@", ""));
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

  private async openSymbolPick() {
    this.symbolPick = new SymbolSelectionQuickPick(this.editor);
    const symbol = await this.symbolPick.start();
    this.quickPick.show();
    if (symbol) {
      this.updateQuickPickValue(this.quickPick.value + `${symbol.label} `);
      this.fileContextLabelToUriMap.set(symbol.label, symbol.uri);
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

    const parseResult = parseUserCommand(command);
    const mentionTexts = parseResult.mentions?.map((mention) => mention.text) || [];
    const uniqueMentionTexts = Array.from(new Set(mentionTexts));

    const userCommand = {
      command,
      context: uniqueMentionTexts
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
  private resultDeffer = new Deferred<SymbolSelectionResult | undefined>();

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
    return this.resultDeffer.promise;
  }

  private async updateSymbolList(query: string) {
    this.quickPick.busy = true;
    const symbolList = await this.fetchSymbolList(query);
    this.quickPick.items = symbolList;
    this.quickPick.busy = false;
  }

  private handleAccept() {
    const selection = this.quickPick.selectedItems[0];
    this.resultDeffer.resolve(
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
    this.resultDeffer.resolve(undefined);
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
      console.error("Error fetching symbols:", error);
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
