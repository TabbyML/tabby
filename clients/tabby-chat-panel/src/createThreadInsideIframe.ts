/* eslint-disable no-console */
/* eslint-disable style/brace-style */
/* eslint-disable antfu/if-newline */
/* eslint-disable style/comma-dangle */
/* eslint-disable style/semi */
/* eslint-disable style/member-delimiter-style */
/* eslint-disable style/quotes */
/* eslint-disable no-restricted-globals */
/* eslint-disable unused-imports/no-unused-vars */
import type { ThreadOptions } from "@quilted/threads";
import { createCustomThread } from "./createThread";
import { createThread } from ".";

const CALL = 0;
const RESULT = 1;
const TERMINATE = 2;
const RELEASE = 3;
const FUNCTION_APPLY = 5;
const FUNCTION_RESULT = 6;
export const CHECK_MESSAGE = "quilt.threads.ping";
export const RESPONSE_MESSAGE = "quilt.threads.pong";
const METHOD_UNAVAILABLE = 7;

export function createThreadFromInsideIframe<
  Self = Record<string, never>,
  Target = Record<string, never>,
>({
  targetOrigin = "*",
  onMethodUnavailable,
  ...options
}: ThreadOptions<Self, Target> & {
  targetOrigin?: string;
  onMethodUnavailable?: (method: string) => void;
} = {}) {
  if (typeof self === "undefined" || self.parent == null) {
    throw new Error(
      "You are not inside an iframe, because there is no parent window."
    );
  }

  const { parent } = self;
  const abort = new AbortController();
  const unavailableMethods = new Set<string>();

  function createCustomListener(originalListener: (data: any) => void) {
    return (event: MessageEvent) => {
      const data = event.data;
      if (Array.isArray(data) && data[0] === METHOD_UNAVAILABLE) {
        const [_, methodName] = data[1];
        console.log(`Method ${methodName} is not available in the iframe.`);
        if (!unavailableMethods.has(methodName)) {
          unavailableMethods.add(methodName);
          onMethodUnavailable?.(methodName);
        }
        return;
      }
      if (event.data !== CHECK_MESSAGE) {
        originalListener(event.data);
      }
    };
  }

  const thread = createThread(
    {
      send(message, transfer) {
        return parent.postMessage(message, targetOrigin, transfer);
      },
      listen(listen, { signal }) {
        const customListener = createCustomListener(listen);
        self.addEventListener("message", customListener, { signal });
      },
    },
    {
      ...options,
    }
  );

  const ready = () => {
    const respond = () => parent.postMessage(RESPONSE_MESSAGE, targetOrigin);
    self.addEventListener(
      "message",
      ({ data }) => {
        if (data === CHECK_MESSAGE) respond();
      },
      { signal: options.signal }
    );
    respond();
  };

  if (document.readyState === "complete") {
    ready();
  } else {
    document.addEventListener(
      "readystatechange",
      () => {
        if (document.readyState === "complete") {
          ready();
          abort.abort();
        }
      },
      { signal: abort.signal }
    );
  }

  return thread;
}
