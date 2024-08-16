import type { Connection } from "vscode-languageserver";
import type { ClientCapabilities, ServerCapabilities } from "./protocol";

export interface Feature {
  setup(
    connection: Connection,
    clientCapabilities: ClientCapabilities,
  ): ServerCapabilities | Promise<ServerCapabilities>;
}
