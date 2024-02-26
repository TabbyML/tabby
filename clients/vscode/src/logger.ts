import { window, LogOutputChannel } from "vscode";

let instance: LogOutputChannel | undefined = undefined;

export function logger(): LogOutputChannel {
  if (!instance) {
    instance = window.createOutputChannel("Tabby", { log: true });
  }
  return instance;
}
