// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import { ExtensionContext, languages } from "vscode";
import { tabbyCommands } from "./Commands";
import { TabbyCompletionProvider } from "./TabbyCompletionProvider";
import { tabbyStatusBarItem } from "./TabbyStatusBarItem";

// this method is called when your extension is activated
// your extension is activated the very first time the command is executed
export function activate(context: ExtensionContext) {
  console.debug("Activating Tabby extension", new Date());
  context.subscriptions.push(
    languages.registerInlineCompletionItemProvider(
      { pattern: "**" },
      new TabbyCompletionProvider()
    ),
    tabbyStatusBarItem,
    ...tabbyCommands
  );
}

// this method is called when your extension is deactivated
export function deactivate() {
  console.debug("Deactivating Tabby extension", new Date());
}
