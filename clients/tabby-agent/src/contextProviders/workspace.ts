import type { Connection } from "vscode-languageserver";
import type { Feature } from "../feature";
import { ClientCapabilities, ServerCapabilities } from "../protocol";

export interface WorkspaceContext {
  uri?: string;
}

export class WorkspaceContextProvider implements Feature {
  // FIXME: add cache and listen to workspace changes

  private lspConnection: Connection | undefined = undefined;
  private clientCapabilities: ClientCapabilities | undefined = undefined;

  constructor() {}

  initialize(connection: Connection, clientCapabilities: ClientCapabilities): ServerCapabilities {
    this.lspConnection = connection;
    this.clientCapabilities = clientCapabilities;
    return {};
  }

  async getWorkspaceContext(uri: string): Promise<WorkspaceContext | undefined> {
    if (this.lspConnection && this.clientCapabilities?.workspace) {
      const workspaceFolders = await this.lspConnection.workspace.getWorkspaceFolders();
      const workspace = workspaceFolders?.find((folder) => uri.startsWith(folder.uri));
      if (workspace) {
        return {
          uri: workspace.uri,
        };
      }
    }
    return undefined;
  }
}
