import * as vscode from "vscode";
import * as os from "os";

interface IdeInfo {
  name: string;
  version: string;
  extensionName: string;
  extensionPublisher: string;
  extensionVersion: string;
  osType: string;
  osRelease: string;
  platform: string;
  arch: string;
}

export function getCurrentIdeInfo(): string {
  const extension = vscode.extensions.getExtension("TabbyML.vscode-tabby");

  const ideInfo: IdeInfo = {
    name: "Visual Studio Code",
    version: vscode.version,
    extensionName: "Tabby",
    extensionPublisher: "TabbyML",
    extensionVersion: extension ? extension.packageJSON.version : "1.8.0-dev", // 使用默认版本，如果无法获取
    osType: os.type(),
    osRelease: os.release(),
    platform: process.platform,
    arch: process.arch,
  };

  return JSON.stringify(ideInfo);
}
