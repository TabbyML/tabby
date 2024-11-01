import { window, LogOutputChannel as VSCodeLogOutputChannel } from "vscode";

const outputChannel = window.createOutputChannel("Tabby", { log: true });

export type LogLevel = "trace" | "debug" | "info" | "warn" | "error";
export interface LogOutputChannel extends VSCodeLogOutputChannel {
  logEveryN(identifier: string, n: number, level: LogLevel, message: string, ...args: unknown[]): void;
}

function tagMessage(message: string, tag: string): string {
  return `[${tag}] ${message}`;
}

export function getLogger(tag = "Tabby"): LogOutputChannel {
  const logEveryNCounts = new Map<string, number>();
  return new Proxy(outputChannel, {
    get(target, method) {
      if (method === "logEveryN") {
        return (identifier: string, n: number, level: LogLevel, message: string, ...args: unknown[]) => {
          const count = logEveryNCounts.get(identifier) ?? 0;
          logEveryNCounts.set(identifier, count + 1);
          if (count % n === 0) {
            target[level](tagMessage(message, tag), ...args);
          }
        };
      }
      if (typeof method == "string" && ["trace", "debug", "info", "warn", "error"].includes(method)) {
        return (message: string, ...args: unknown[]) => {
          /* @ts-expect-error no-implicit-any */
          target[method]?.(tagMessage(message, tag), ...args);
        };
      }
      if (method in target) {
        /* @ts-expect-error no-implicit-any */
        return target[method];
      }
      return undefined;
    },
  }) as LogOutputChannel;
}

export function showOutputPanel(): void {
  outputChannel.show();
}
