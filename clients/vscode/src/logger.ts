import { window, LogOutputChannel } from "vscode";

const outputChannel = window.createOutputChannel("Tabby", { log: true });

interface Throttled {
  key: string;
  updatedAt: number;
  repeat: number;
}

export function getLogger(tag = "Tabby"): LogOutputChannel {
  const tagMessage = (message: string, repeat: number | undefined = undefined) => {
    const repeatMessage = repeat && repeat > 1 ? `(x${repeat}) ` : "";
    return `[${tag}] ${repeatMessage}${message}`;
  };

  let throttled: Throttled | null = null;
  const throttleInterval = 10 * 1000; // 10s
  const throttle = (key: string, callback: (repeat?: number | undefined) => void) => {
    const now = Date.now();
    if (throttled && throttled.key === key) {
      throttled.repeat++;
      if (now - throttled.updatedAt < throttleInterval) {
        return;
      } else {
        throttled.updatedAt = now;
        callback(throttled.repeat);
      }
    } else {
      throttled = { key, updatedAt: now, repeat: 1 };
      callback();
    }
  };

  return new Proxy(outputChannel, {
    get(target, method) {
      if (typeof method == "string" && ["trace", "debug", "info", "warn", "error"].includes(method)) {
        return (message: string, ...args: unknown[]) => {
          const key = JSON.stringify({
            method,
            tag,
            message,
          });
          throttle(key, (repeat: number | undefined) => {
            const taggedMessage = tagMessage(message, repeat);
            /* @ts-expect-error no-implicit-any */
            target[method]?.(taggedMessage, ...args);
          });
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

export function showOutputPanel(): void {
  outputChannel.show();
}
