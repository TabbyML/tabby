import { window, ExtensionContext, Uri } from "vscode";
import { LanguageClientOptions } from "vscode-languageclient";
import { LanguageClient as NodeLanguageClient, ServerOptions, TransportKind } from "vscode-languageclient/node";
import { LanguageClient as BrowserLanguageClient } from "vscode-languageclient/browser";
import { getLogger } from "./logger";
import { Client } from "./lsp/Client";
import { InlineCompletionProvider } from "./InlineCompletionProvider";
import { Config } from "./Config";
import { Issues } from "./Issues";
import { GitProvider } from "./git/GitProvider";
import { ContextVariables } from "./ContextVariables";
import { StatusBarItem } from "./StatusBarItem";
import { ChatViewProvider } from "./chat/ChatViewProvider";
import { Commands } from "./Commands";

const isBrowser = !!process.env["IS_BROWSER"];
const logger = getLogger();
let client: Client | undefined = undefined;

export async function activate(context: ExtensionContext) {
  logger.info("Activating Tabby extension...");
  const clientOptions: LanguageClientOptions = {
    documentSelector: [
      { scheme: "file" },
      { scheme: "untitled" },
      { scheme: "vscode-notebook-cell" },
      { scheme: "vscode-userdata" },
    ],
    outputChannel: logger,
  };
  if (isBrowser) {
    const workerModulePath = Uri.joinPath(context.extensionUri, "dist/tabby-agent/browser/index.mjs");
    const worker = new Worker(workerModulePath.toString());
    const languageClient = new BrowserLanguageClient("Tabby", "Tabby", clientOptions, worker);
    client = new Client(context, languageClient);
  } else {
    const serverModulePath = context.asAbsolutePath("dist/tabby-agent/node/index.js");
    const serverOptions: ServerOptions = {
      run: {
        module: serverModulePath,
        transport: TransportKind.ipc,
      },
      debug: {
        module: serverModulePath,
        transport: TransportKind.ipc,
      },
    };
    const languageClient = new NodeLanguageClient("Tabby", serverOptions, clientOptions);
    client = new Client(context, languageClient);
  }
  // Register chat panel
  const chatViewProvider = new ChatViewProvider(context, client.agent, logger);
  context.subscriptions.push(
    window.registerWebviewViewProvider("tabby.chatView", chatViewProvider, {
      webviewOptions: { retainContextWhenHidden: true },
    }),
  );

  const config = new Config(context);
  const inlineCompletionProvider = new InlineCompletionProvider(client, config);
  const gitProvider = new GitProvider();
  client.registerConfigManager(config);
  client.registerInlineCompletionProvider(inlineCompletionProvider);
  client.registerGitProvider(gitProvider);

  await client.start();

  const issues = new Issues(client, config);
  const contextVariables = new ContextVariables(client, config);
  /* eslint-disable-next-line @typescript-eslint/no-unused-vars */ /* @ts-expect-error noUnusedLocals */
  const statusBarItem = new StatusBarItem(context, client, config, issues, inlineCompletionProvider);
  /* eslint-disable-next-line @typescript-eslint/no-unused-vars */ /* @ts-expect-error noUnusedLocals */
  const commands = new Commands(
    context,
    client,
    config,
    contextVariables,
    inlineCompletionProvider,
    chatViewProvider,
    gitProvider,
  );

  logger.info("Tabby extension activated.");
}

export async function deactivate() {
  logger.info("Deactivating Tabby extension...");
  await client?.stop();
  logger.info("Tabby extension deactivated.");
}
