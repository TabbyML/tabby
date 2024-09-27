import type { Connection } from "vscode-languageserver";
import type { ClientCapabilities, ClientProvidedConfig, ServerCapabilities } from "./protocol";

export interface Feature {
  initialize(
    connection: Connection,
    clientCapabilities: ClientCapabilities,
    clientProvidedConfig: ClientProvidedConfig,
  ): ServerCapabilities | Promise<ServerCapabilities>;

  initialized?(connection: Connection): void | Promise<void>;

  shutdown?(): void | Promise<void>;
}
