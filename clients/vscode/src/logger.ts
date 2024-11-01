import { window, LogOutputChannel as VSCodeLogOutputChannel } from "vscode";

const outputChannel = window.createOutputChannel("Tabby", { log: true });

export type LogLevel = "trace" | "debug" | "info" | "warn" | "error";

export interface LogEveryNOptions {
  identifier?: string;
  every?: number;
  level?: LogLevel;
}

export interface LogOutputChannel extends VSCodeLogOutputChannel {
  log(message: string, ...args: unknown[]): void;
  log(options: LogEveryNOptions, message: string, ...args: unknown[]): void;
}

function tagMessage(message: string, tag: string): string {
  return `[${tag}] ${message}`;
}

export function getLogger(tag = "Tabby"): LogOutputChannel {
  const logEveryNCounts = new Map<string, number>();
  return new Proxy(outputChannel, {
    get(target, method) {
      if (method === "log") {
        return (...args: unknown[]) => {
          let options: LogEveryNOptions = {};
          let message: string;
          if (typeof args[0] === "string") {
            message = args.shift() as string;
          } else if (typeof args[0] === "object") {
            options = args.shift() as LogEveryNOptions;
            message = args.shift() as string;
          } else {
            return;
          }
          const { identifier = message, every = 1, level = "info" } = options;
          const count = logEveryNCounts.get(identifier) ?? 0;
          logEveryNCounts.set(identifier, count + 1);
          if (count % every === 0) {
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
