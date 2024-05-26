import { commands } from "vscode";
import { Client } from "./lsp/Client";
import { Config } from "./Config";

export class ContextVariables {
  private chatEnabledValue = false;
  private explainCodeBlockEnabledValue = false;
  private generateCommitMessageEnabledValue = false;

  constructor(
    private readonly client: Client,
    private readonly config: Config,
  ) {
    this.chatEnabled = this.client.chat.isAvailable;
    this.client.chat.on("didChangeAvailability", (params: boolean) => {
      this.chatEnabled = params;
    });
    this.updateExperimentalFlags();
    this.config.on("updated", () => {
      this.updateExperimentalFlags();
    });
  }

  private updateExperimentalFlags() {
    const experimental = this.config.workspace.get<Record<string, unknown>>("experimental", {});
    this.explainCodeBlockEnabled = !!experimental["chat.explainCodeBlock"];
    this.generateCommitMessageEnabled = !!experimental["chat.generateCommitMessage"];
  }

  get chatEnabled(): boolean {
    return this.chatEnabledValue;
  }

  private set chatEnabled(value: boolean) {
    commands.executeCommand("setContext", "tabby.chatEnabled", value);
    this.chatEnabledValue = value;
  }

  get explainCodeBlockEnabled(): boolean {
    return this.explainCodeBlockEnabledValue;
  }

  private set explainCodeBlockEnabled(value: boolean) {
    commands.executeCommand("setContext", "tabby.explainCodeBlockEnabled", value);
    this.explainCodeBlockEnabledValue = value;
  }

  get generateCommitMessageEnabled(): boolean {
    return this.generateCommitMessageEnabledValue;
  }

  private set generateCommitMessageEnabled(value: boolean) {
    commands.executeCommand("setContext", "tabby.generateCommitMessageEnabled", value);
    this.generateCommitMessageEnabledValue = value;
  }
}
