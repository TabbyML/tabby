import type { CancellationToken, Connection } from "vscode-languageserver";
import type { Feature } from "../feature";
import { ClientCapabilities, ServerCapabilities, EditorOptionsRequest, EditorOptions } from "../protocol";

export type EditorOptionsContext = EditorOptions;

export class EditorOptionsProvider implements Feature {
  // FIXME: add cache and listen to editor options changes

  private lspConnection: Connection | undefined = undefined;
  private clientCapabilities: ClientCapabilities | undefined = undefined;

  constructor() {}

  initialize(connection: Connection, clientCapabilities: ClientCapabilities): ServerCapabilities {
    this.lspConnection = connection;
    this.clientCapabilities = clientCapabilities;
    return {};
  }

  async getEditorOptions(uri: string, token: CancellationToken): Promise<EditorOptionsContext | undefined> {
    if (this.lspConnection && this.clientCapabilities?.tabby?.editorOptions) {
      const editorOptions: EditorOptions | null = await this.lspConnection.sendRequest(
        EditorOptionsRequest.type,
        {
          uri,
        },
        token,
      );
      if (editorOptions) {
        return editorOptions;
      }
    }
    return undefined;
  }
}
