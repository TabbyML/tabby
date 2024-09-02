import { commands, window, workspace, Range } from "vscode";
import { Client } from "./lsp/Client";
import { Config } from "./Config";

export class ContextVariables {
  private chatEnabledValue = false;
  private chatEditInProgressValue = false;
  private chatEditResolvingValue = false;
  private inlineCompletionTriggerModeValue: "automatic" | "manual" = "automatic";

  constructor(
    private readonly client: Client,
    private readonly config: Config,
  ) {
    this.chatEnabled = this.client.chat.isAvailable;
    this.inlineCompletionTriggerMode = config.inlineCompletionTriggerMode;
    this.client.chat.on("didChangeAvailability", (params: boolean) => {
      this.chatEnabled = params;
    });
    this.config.on("updated", () => {
      this.inlineCompletionTriggerMode = config.inlineCompletionTriggerMode;
    });
    this.updateChatEditResolving();
    window.onDidChangeTextEditorSelection((params) => {
      if (params.textEditor === window.activeTextEditor) {
        this.updateChatEditResolving();
      }
    });
    workspace.onDidChangeTextDocument((params) => {
      if (params.document === window.activeTextEditor?.document) {
        this.updateChatEditResolving();
      }
    });
  }

  updateChatEditResolving() {
    const editor = window.activeTextEditor;
    if (!editor) {
      this.chatEditResolving = false;
      return;
    }
    const range = new Range(editor.selection.active.line, 0, editor.selection.active.line + 1, 0);
    const text = editor.document.getText(range);
    const match = /^<<<<<<< (tabby-[0-9|a-z|A-Z]{6})/g.exec(text);
    if (match) {
      this.chatEditResolving = true;
      return;
    }
    this.chatEditResolving = false;
  }

  get chatEnabled(): boolean {
    return this.chatEnabledValue;
  }

  private set chatEnabled(value: boolean) {
    commands.executeCommand("setContext", "tabby.chatEnabled", value);
    this.chatEnabledValue = value;
  }

  get chatEditInProgress(): boolean {
    return this.chatEditInProgressValue;
  }

  set chatEditInProgress(value: boolean) {
    commands.executeCommand("setContext", "tabby.chatEditInProgress", value);
    this.chatEditInProgressValue = value;
  }

  get chatEditResolving(): boolean {
    return this.chatEditResolvingValue;
  }

  set chatEditResolving(value: boolean) {
    commands.executeCommand("setContext", "tabby.chatEditResolving", value);
    this.chatEditResolvingValue = value;
  }

  get inlineCompletionTriggerMode(): "automatic" | "manual" {
    return this.inlineCompletionTriggerModeValue;
  }

  set inlineCompletionTriggerMode(value: "automatic" | "manual") {
    commands.executeCommand("setContext", "tabby.inlineCompletionTriggerMode", value);
    this.inlineCompletionTriggerModeValue = value;
  }
}
