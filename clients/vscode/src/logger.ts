import { window, LogOutputChannel } from "vscode";

const outputChannel = window.createOutputChannel("Tabby", { log: true });

export function getLogger(tag = "Tabby"): LogOutputChannel {
  const tagMessage = (message: string) => {
    return `[${tag}] ${message}`;
  };
  return new Proxy(outputChannel, {
    get(target, method) {
      if (method === "trace") {
        return (message: string, ...args: unknown[]) => {
          target.trace(tagMessage(message), ...args);
        };
      }
      if (method === "debug") {
        return (message: string, ...args: unknown[]) => {
          target.debug(tagMessage(message), ...args);
        };
      }
      if (method === "info") {
        return (message: string, ...args: unknown[]) => {
          target.info(tagMessage(message), ...args);
        };
      }
      if (method === "warn") {
        return (message: string, ...args: unknown[]) => {
          target.warn(tagMessage(message), ...args);
        };
      }
      if (method === "error") {
        return (message: string, ...args: unknown[]) => {
          target.error(tagMessage(message), ...args);
        };
      }
      if (method in target) {
        /* @ts-expect-error no-implicit-any */
        return target[method];
      }
      return undefined;
    },
  });
}
