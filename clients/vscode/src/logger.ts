import { window, LogOutputChannel } from "vscode";

const outputChannel = window.createOutputChannel("Tabby", { log: true });

function tagMessage(message: string, tag: string): string {
  return `[${tag}] ${message}`;
}

export function getLogger(tag = "Tabby"): LogOutputChannel {
  return new Proxy(outputChannel, {
    get(target, method) {
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
  });
}

export function getLoggerEveryN(
  n: number,
  level: "trace" | "debug" | "info" | "warn" | "error",
  tag = "Tabby",
): (message: string, ...args: unknown[]) => void {
  const doLog = getLogger(tag)[level];
  let count = 0;
  return (message: string, ...args: unknown[]) => {
    if (count % n === 0) {
      doLog(message, ...args);
    }
    count++;
  };
}

export function showOutputPanel(): void {
  outputChannel.show();
}
