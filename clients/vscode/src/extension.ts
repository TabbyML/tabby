// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import { ExtensionContext, languages, workspace } from "vscode";
import { logger } from "./logger";
import { createAgentInstance, disposeAgentInstance } from "./agent";
import { tabbyCommands } from "./commands";
import { TabbyCompletionProvider } from "./TabbyCompletionProvider";
import { TabbyStatusBarItem } from "./TabbyStatusBarItem";
import { RecentlyChangedCodeSearch } from "./RecentlyChangedCodeSearch";

// this method is called when your extension is activated
// your extension is activated the very first time the command is executed
export async function activate(context: ExtensionContext) {
  logger().info("Activating Tabby extension");
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
}

// this method is called when your extension is deactivated
export async function deactivate() {
  logger().info("Deactivating Tabby extension");
  await disposeAgentInstance();
}
