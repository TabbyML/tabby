// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import { ExtensionContext, commands, languages, workspace } from "vscode";
import { getLogger } from "./logger";
import { createAgentInstance, disposeAgentInstance } from "./agent";
import { tabbyCommands } from "./commands";
import { TabbyCompletionProvider } from "./TabbyCompletionProvider";
import { TabbyStatusBarItem } from "./TabbyStatusBarItem";
import { RecentlyChangedCodeSearch } from "./RecentlyChangedCodeSearch";

const logger = getLogger();

// this method is called when your extension is activated
// your extension is activated the very first time the command is executed
export async function activate(context: ExtensionContext) {
  logger.info("Activating Tabby extension...");
  const agent = await createAgentInstance(context);
  const completionProvider = new TabbyCompletionProvider();
  context.subscriptions.push(languages.registerInlineCompletionItemProvider({ pattern: "**" }, completionProvider));

  const collectSnippetsFromRecentChangedFilesConfig =
    agent.getConfig().completion.prompt.collectSnippetsFromRecentChangedFiles;
  if (collectSnippetsFromRecentChangedFilesConfig.enabled) {
    const recentlyChangedCodeSnippetsIndex = new RecentlyChangedCodeSearch(
      collectSnippetsFromRecentChangedFilesConfig.indexing,
    );
    context.subscriptions.push(
      workspace.onDidChangeTextDocument((event) => {
        // Ensure that the changed file is belong to a workspace folder
        const workspaceFolder = workspace.getWorkspaceFolder(event.document.uri);
        if (workspaceFolder && workspace.workspaceFolders?.includes(workspaceFolder)) {
          recentlyChangedCodeSnippetsIndex.handleDidChangeTextDocument(event);
        }
      }),
    );
    completionProvider.recentlyChangedCodeSearch = recentlyChangedCodeSnippetsIndex;
  }

  const statusBarItem = new TabbyStatusBarItem(context, completionProvider);
  context.subscriptions.push(statusBarItem.register());

  context.subscriptions.push(...tabbyCommands(context, completionProvider, statusBarItem));

  const updateIsChatEnabledContextVariable = () => {
    if (agent.getStatus() === "ready") {
      const healthState = agent.getServerHealthState();
      const isChatEnabled = Boolean(healthState?.chat_model);
      commands.executeCommand("setContext", "tabby.chatModeEnabled", isChatEnabled);
    }
  };
  agent.on("statusChanged", () => {
    updateIsChatEnabledContextVariable();
  });
  updateIsChatEnabledContextVariable();

  const updateIsExplainCodeEnabledContextVariable = () => {
    const experimental = workspace.getConfiguration("tabby").get<Record<string, any>>("experimental", {});
    const isExplainCodeEnabled = experimental["chat.explainCodeBlock"] || false;
    commands.executeCommand("setContext", "tabby.explainCodeSettingEnabled", isExplainCodeEnabled);
  };
  workspace.onDidChangeConfiguration((event) => {
    if (event.affectsConfiguration("tabby.experimental")) {
      updateIsExplainCodeEnabledContextVariable();
    }
  });
  updateIsExplainCodeEnabledContextVariable();
  logger.info("Tabby extension activated.");
}

// this method is called when your extension is deactivated
export async function deactivate() {
  logger.info("Deactivating Tabby extension...");
  await disposeAgentInstance();
  logger.info("Tabby extension deactivated.");
}
