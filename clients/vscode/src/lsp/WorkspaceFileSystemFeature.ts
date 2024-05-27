import { workspace, Range, Uri } from "vscode";
import { BaseLanguageClient, StaticFeature, FeatureState, Disposable } from "vscode-languageclient";
import { ClientCapabilities, ReadFileRequest, ReadFileParams } from "tabby-agent";

export class WorkspaceFileSystemFeature implements StaticFeature {
  private disposables: Disposable[] = [];

  constructor(private readonly client: BaseLanguageClient) {}

  getState(): FeatureState {
    return { kind: "static" };
  }

  fillInitializeParams() {
    // nothing
  }

  fillClientCapabilities(capabilities: ClientCapabilities): void {
    capabilities.tabby = {
      ...capabilities.tabby,
      workspaceFileSystem: true,
    };
  }

  preInitialize(): void {
    // nothing
  }

  initialize(): void {
    this.disposables.push(
      this.client.onRequest(ReadFileRequest.type, async (params: ReadFileParams) => {
        if (params.format !== "text") {
          return null;
        }
        const textDocument = await workspace.openTextDocument(Uri.parse(params.uri));
        const range = params.range
          ? new Range(
              params.range.start.line,
              params.range.start.character,
              params.range.end.line,
              params.range.end.character,
            )
          : undefined;
        return {
          text: textDocument.getText(range),
        };
      }),
    );
  }

  clear(): void {
    this.disposables.forEach((disposable) => disposable.dispose());
    this.disposables = [];
  }
}
