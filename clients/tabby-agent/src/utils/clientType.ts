import type { ClientInfo } from "../protocol";

export type ClientType = "vscode" | "intellij" | "vim" | "unknown";

export function getClientType(clientInfo: ClientInfo | undefined): ClientType {
  const pluginName = clientInfo?.tabbyPlugin?.name ?? "";
  if (pluginName.includes("vscode")) {
    return "vscode";
  }
  if (pluginName.includes("intellij")) {
    return "intellij";
  }
  if (pluginName.includes("vim")) {
    return "vim";
  }
  return "unknown";
}
