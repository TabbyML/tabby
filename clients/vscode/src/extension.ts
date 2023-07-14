// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import { ExtensionContext, languages } from "vscode";
import { createAgentInstance } from "./agent";
import { tabbyCommands } from "./commands";
import { TabbyCompletionProvider } from "./TabbyCompletionProvider";
import { tabbyStatusBarItem } from "./statusBarItem";

// this method is called when your extension is activated
// your extension is activated the very first time the command is executed
export async function activate(context: ExtensionContext) {
  console.debug("Activating Tabby extension", new Date());
  await createAgentInstance(context);
  context.subscriptions.push(
    languages.registerInlineCompletionItemProvider({ pattern: "**" }, new TabbyCompletionProvider()),
    tabbyStatusBarItem(),
    ...tabbyCommands(),
  );
}

// this method is called when your extension is deactivated
export function deactivate() {
  console.debug("Deactivating Tabby extension", new Date());
}
