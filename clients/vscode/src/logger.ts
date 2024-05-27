import { window, LogOutputChannel } from "vscode";

let instance: LogOutputChannel | undefined = undefined;

export function getLogChannel(): LogOutputChannel {
  if (!instance) {
    instance = window.createOutputChannel("Tabby", { log: true });
  }
  return instance;
}

export function getLogger(tag: string = "Tabby"): LogOutputChannel {
  const rawLogger = getLogChannel();
  const tagMessage = (message: string) => {
    return `[${tag}] ${message}`;
  };
  return {
    ...rawLogger,
    trace: (message: string, ...args: any[]) => {
      rawLogger.trace(tagMessage(message), ...args);
    },
    debug: (message: string, ...args: any[]) => {
      rawLogger.debug(tagMessage(message), ...args);
    },
    info: (message: string, ...args: any[]) => {
      rawLogger.info(tagMessage(message), ...args);
    },
    warn: (message: string, ...args: any[]) => {
      rawLogger.warn(tagMessage(message), ...args);
    },
    error: (message: string, ...args: any[]) => {
      rawLogger.error(tagMessage(message), ...args);
    },
  };
}
