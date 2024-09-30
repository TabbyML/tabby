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
import { ChatSideViewProvider } from "./chat/ChatSideViewProvider";
import { Commands } from "./Commands";
import { Status } from "tabby-agent";
import { CodeActions } from "./CodeActions";
import { isBrowser } from "./env";

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
  const config = new Config(context);
  const contextVariables = new ContextVariables(client, config);
  const inlineCompletionProvider = new InlineCompletionProvider(client, config);
  const gitProvider = new GitProvider();
  client.registerConfigManager(config);
  client.registerInlineCompletionProvider(inlineCompletionProvider);
  client.registerGitProvider(gitProvider);

  // Register config callback for past ServerConfig
  client.agent.addListener("didChangeStatus", async (status: Status) => {
    if (!client) return;

    const { config: serverConfig } = await client.agent.fetchServerInfo();

    if (serverConfig.requestHeaders && Object.keys(serverConfig.requestHeaders).length > 0) {
      // If serverConfig.requestHeaders is not empty, it means the server is configured in `tabby-agent/config.toml`, we shall not record it.
      return;
    }

    if (status === "ready") {
      await config.appendPastServerConfig({
        endpoint: serverConfig.endpoint,
        token: serverConfig.token,
      });
    }
  });

  // Register chat panel
  const chatViewProvider = new ChatSideViewProvider(context, client.agent, logger, gitProvider);
  context.subscriptions.push(
    window.registerWebviewViewProvider("tabby.chatView", chatViewProvider, {
      webviewOptions: { retainContextWhenHidden: true },
    }),
  );
  // Create chat panel view
  await gitProvider.init();
  await client.start();

  const issues = new Issues(client, config);
  /* eslint-disable-next-line @typescript-eslint/ban-ts-comment */ /* eslint-disable-next-line @typescript-eslint/prefer-ts-expect-error */
  /* eslint-disable-next-line @typescript-eslint/no-unused-vars */ // @ts-ignore noUnusedLocals
  const statusBarItem = new StatusBarItem(context, client, config, issues, inlineCompletionProvider);
  /* eslint-disable-next-line @typescript-eslint/ban-ts-comment */ /* eslint-disable-next-line @typescript-eslint/prefer-ts-expect-error */
  /* eslint-disable-next-line @typescript-eslint/no-unused-vars */ // @ts-ignore noUnusedLocals
  const commands = new Commands(
    context,
    client,
    config,
    issues,
    contextVariables,
    inlineCompletionProvider,
    chatViewProvider,
    gitProvider,
  );
  /* eslint-disable-next-line @typescript-eslint/ban-ts-comment */ /* eslint-disable-next-line @typescript-eslint/prefer-ts-expect-error */
  /* eslint-disable-next-line @typescript-eslint/no-unused-vars */ // @ts-ignore noUnusedLocals
  const codeActions = new CodeActions(client, contextVariables);

  logger.info("Tabby extension activated.");
}

export async function deactivate() {
  logger.info("Deactivating Tabby extension...");
  await client?.stop();
  logger.info("Tabby extension deactivated.");
}
