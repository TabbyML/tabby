import { ExtensionContext } from "vscode";
import { LanguageClient, ServerOptions, TransportKind } from "vscode-languageclient/node";

let client: LanguageClient;

export async function activate(context: ExtensionContext): Promise<void> {
  console.debug("Tabby LSP Example: activate");

  const serverModulePath = context.asAbsolutePath("dist/server/tabby-agent.js");
  const serverOptions: ServerOptions = {
    run: {
      module: serverModulePath,
      args: ["--lsp"],
      transport: TransportKind.ipc,
    },
    debug: {
      module: serverModulePath,
      args: ["--lsp"],
      transport: TransportKind.ipc,
    },
  };
  const clientOptions = {
    documentSelector: [{ pattern: "**/*" }],
  };
  if (!client) {
    client = new LanguageClient("Tabby LSP Example", serverOptions, clientOptions);
  }
  return await client.start();
}

export async function deactivate(): Promise<void> {
  console.debug("Tabby LSP Example: deactivate");

  if (client) {
    return await client.stop();
  }
}
