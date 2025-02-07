import { window, ExtensionContext } from "vscode";
import { getLogger } from "./logger";
import { Client, createClient } from "./lsp/client";
import { InlineCompletionProvider } from "./InlineCompletionProvider";
import { Config } from "./Config";
import { GitProvider } from "./git/GitProvider";
import { ContextVariables } from "./ContextVariables";
import { StatusBarItem } from "./StatusBarItem";
import { ChatSidePanelProvider } from "./chat/sidePanel";
import { Commands } from "./commands";
import { init as initFindFiles } from "./findFiles";
import { CodeActions } from "./CodeActions";

const logger = getLogger();
let clientRef: Client | undefined = undefined;

export async function activate(context: ExtensionContext) {
  logger.info("Activating Tabby extension...");

  const client = createClient(context, logger);
  const config = new Config(context);
  const contextVariables = new ContextVariables(client, config);
  const inlineCompletionProvider = new InlineCompletionProvider(client, config);
  const gitProvider = new GitProvider();

  client.registerConfigManager(config);
  client.registerInlineCompletionProvider(inlineCompletionProvider);
  client.registerGitProvider(gitProvider);
  clientRef = client;

  // Register chat panel
  const chatViewProvider = new ChatSidePanelProvider(context, client, contextVariables, gitProvider);
  context.subscriptions.push(
    window.registerWebviewViewProvider("tabby.chatView", chatViewProvider, {
      webviewOptions: { retainContextWhenHidden: true },
    }),
  );

  // Register status bar item
  const statusBarItem = new StatusBarItem(client, config);
  statusBarItem.registerInContext(context);

  // Register command
  const commands = new Commands(
    context,
    client,
    config,
    contextVariables,
    inlineCompletionProvider,
    chatViewProvider,
    gitProvider,
  );
  commands.register();

  // Register code actions
  /* eslint-disable-next-line @typescript-eslint/ban-ts-comment */ /* eslint-disable-next-line @typescript-eslint/prefer-ts-expect-error */
  /* eslint-disable-next-line @typescript-eslint/no-unused-vars */ // @ts-ignore noUnusedLocals
  const codeActions = new CodeActions(client, contextVariables);

  logger.info("Tabby extension activated.");

  // Start async initialization
  const startClient = async () => {
    await gitProvider.init();

    logger.info("Launching language server tabby-agent...");
    await client.start();
    logger.info("Language server tabby-agent launched.");
  };

  await Promise.all([
    // start LSP client
    startClient(),

    // findFiles preheat
    initFindFiles(context),
  ]);
}

export async function deactivate() {
  logger.info("Deactivating Tabby extension...");
  await clientRef?.stop();
  logger.info("Tabby extension deactivated.");
}
