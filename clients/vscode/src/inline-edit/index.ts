import { ChatEditCommand } from "tabby-agent";
import { Config } from "../Config";
import { CancellationTokenSource, QuickPickItem, ThemeIcon, QuickPickItemKind, window, ProgressLocation, TextEditor, Selection, Position } from "vscode";
import { Client } from "../lsp/Client";
import { ContextVariables } from "../ContextVariables";

export class InlineEditController {
    private chatEditCancellationTokenSource: CancellationTokenSource | null = null;

    constructor(private client: Client, private config: Config,
        private contextVariables: ContextVariables,
        editor: TextEditor,
        editLocation: EditLocation) {
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

        const startPosition = new Position(editLocation.range.start.line, editLocation.range.start.character);
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
    }
}

interface EditLocation {
    uri: string;
    range: {
        start: { line: number; character: number };
        end: { line: number; character: number };
    };
}