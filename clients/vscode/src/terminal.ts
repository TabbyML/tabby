import { commands, env, window } from "vscode";
import { TerminalContext } from "tabby-chat-panel/index";

export async function getTerminalContext(): Promise<TerminalContext | undefined> {
  const activeTerminal = window.activeTerminal;
  if (!activeTerminal) {
    return;
  }
  const processId = await activeTerminal.processId;

  // Store current clipboard content to restore it later
  const originalClipboardContent = await env.clipboard.readText();
  await commands.executeCommand("workbench.action.terminal.copySelection");
  const selectedText = await env.clipboard.readText();
  await env.clipboard.writeText(originalClipboardContent);

  if (!selectedText || selectedText.trim() === "") {
    return;
  }
  return {
    kind: "terminal",
    name: activeTerminal.name,
    processId: processId,
    selection: selectedText,
  };
}
