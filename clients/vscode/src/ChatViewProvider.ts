import { ExtensionContext, Disposable, Webview, WebviewPanel, window, Uri, ViewColumn, WebviewViewProvider, WebviewView, TextDocument } from "vscode";
import { getUri } from "./utils";

import { createAgentInstance, disposeAgentInstance } from "./agent";

export class ChatViewProvider implements WebviewViewProvider {
  webview?: WebviewView;

  constructor(private readonly context: ExtensionContext) {}

  public async resolveWebviewView(webviewView: WebviewView) {
    this.webview = webviewView;
    const extensionUri = this.context.extensionUri

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [extensionUri],
    };
    webviewView.webview.html = await this._getWebviewContent(webviewView.webview, extensionUri);
    webviewView.webview.onDidReceiveMessage((data) => {
      switch (data.command) {
        case 'explainThis': {
          const { text, language } = data;
          console.log('text', text)
          console.log('language', language)
          return;
        }
      }
    });
  }

  private async _getWebviewContent(webview: Webview, extensionUri: Uri) {
    const agent = await createAgentInstance(this.context);
    const { server } = agent.getConfig()
    const scriptUri = getUri(webview, extensionUri, ['chat-panel', "index.js"]);
    return `
      <!DOCTYPE html>
      <html lang="en">
        <head>
          <meta charset="UTF-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <title>Tabby</title>
        </head>
        <style>
          html, body, iframe {
            padding: 0;
            margin: 0;
            box-sizing: border-box;
            border-width: 0;
          }
        </style>
        <body>
          <div id="root" style="height: 100vh; overflow: hidden"></div>
          <script>window.endpoint="${server.endpoint}"</script>
          <script>window.token="${server.token}"</script>
          <script type="module" src="${scriptUri}"></script>
        </body>
      </html>
    `;
  }

  /**
   * Sets up an event listener to listen for messages passed from the webview context and
   * executes code based on the message that is recieved.
   *
   * @param webview A reference to the extension webview
   * @param context A reference to the extension context
   */
  // private _setWebviewMessageListener(webview: Webview) {
  //   webview.onDidReceiveMessage(
  //     (message: any) => {
  //       const command = message.command;
  //       const text = message.text;

  //       switch (command) {
  //         case "hello":
  //           // Code that should run in response to the hello message command
  //           window.showInformationMessage(text);
  //           return;
  //         // Add more switch case statements here as more webview message commands
  //         // are created within the webview context (i.e. inside media/main.js)
  //       }
  //     },
  //     undefined,
  //     this._disposables
  //   );
  // }

  public revive(panel: WebviewView) {
    this.webview = panel;
  }

  public postMessage (message: any) {
    if (this.webview) {
      console.log('this.webview exist')
      console.log('this.webview?.webview.', this.webview?.webview)
      this.webview?.webview.postMessage(message)
    }
  }
}