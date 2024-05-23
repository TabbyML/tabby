import { createThread, type ThreadOptions } from '@quilted/threads';
import { ServerApi } from 'tabby-chat-panel'
import { WebviewView } from "vscode";

const CALL = 0;

export function createThreadFromWebview(webview: WebviewView, options?: ThreadOptions) {
  return createThread(
    {
      send(...args) {
        webview.webview.postMessage({ data: args });
        console.log('message sent')
      },
      listen(listen, { signal }) {
        webview.webview.onDidReceiveMessage(data => {
          listen(data)
        })
        // signal.addEventListener("abort", () => {
        //   // Abort, remove listener
        //   panel.webview.onDidReceiveMessage(undefined, undefined, context.subscriptions);
        // })
      },
    },
    options,
  );
}