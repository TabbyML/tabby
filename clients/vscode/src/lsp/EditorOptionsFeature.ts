import { Uri } from "vscode";
import { BaseLanguageClient, StaticFeature, FeatureState, Disposable } from "vscode-languageclient";
import { ClientCapabilities, EditorOptionsRequest, EditorOptionsParams } from "tabby-agent";
import { findTextEditor } from "./vscodeWindowUtils";

export class EditorOptionsFeature implements StaticFeature {
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
      editorOptions: true,
    };
  }

  preInitialize(): void {
    // nothing
  }

  initialize(): void {
    this.disposables.push(
      this.client.onRequest(EditorOptionsRequest.type, (params: EditorOptionsParams) => {
        const editor = findTextEditor(Uri.parse(params.uri));
        if (!editor) {
          return null;
        }
        const { insertSpaces, tabSize } = editor.options;
        let indentation: string | undefined;
        if (insertSpaces && typeof tabSize === "number" && tabSize > 0) {
          indentation = " ".repeat(tabSize);
        } else if (!insertSpaces) {
          indentation = "\t";
        }
        return {
          indentation,
        };
      }),
    );
  }

  clear(): void {
    this.disposables.forEach((disposable) => disposable.dispose());
    this.disposables = [];
  }
}
