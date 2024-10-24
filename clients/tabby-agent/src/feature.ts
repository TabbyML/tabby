import type { Connection } from "vscode-languageserver";
import type { ClientCapabilities, ClientProvidedConfig, ServerCapabilities, DataStoreRecords } from "./protocol";

export interface Feature {
  initialize(
    connection: Connection,
    clientCapabilities: ClientCapabilities,
    clientProvidedConfig: ClientProvidedConfig,
    dataStoreRecords: DataStoreRecords | undefined,
  ): ServerCapabilities | Promise<ServerCapabilities>;

  initialized?(connection: Connection): void | Promise<void>;

  shutdown?(): void | Promise<void>;
}
