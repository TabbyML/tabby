import {
  Disposable,
  DocumentUri,
  TextDocumentsConfiguration,
  DidChangeTextDocumentParams,
  TextDocuments as LspTextDocuments,
} from "vscode-languageserver";
import { TextDocumentConnection } from "vscode-languageserver/lib/common/textDocuments";

export class TextDocuments<
  T extends {
    uri: DocumentUri;
  },
> extends LspTextDocuments<T> {
  constructor(configuration: TextDocumentsConfiguration<T>) {
    super(configuration);
  }

  override listen(connection: TextDocumentConnection): Disposable {
    const disposables: Disposable[] = [];
    disposables.push(super.listen(connection));
    // override onDidChangeTextDocument listener
    disposables.push(
      connection.onDidChangeTextDocument((params: DidChangeTextDocumentParams) => {
        const { textDocument, contentChanges } = params;
        if (contentChanges.length === 0) {
          return;
        }
        const { version } = textDocument;
        if (version === null || version === undefined) {
          throw new Error(`Received document change event for ${textDocument.uri} without valid version identifier`);
        }
        let document = this.get(textDocument.uri);
        if (document !== undefined) {
          document = this["_configuration"].update(document, contentChanges, version);
          this["_syncedDocuments"].set(textDocument.uri, document);
          this["_onDidChangeContent"].fire(Object.freeze({ document: document, contentChanges }));
        }
      }),
    );
    return Disposable.create(() => {
      disposables.forEach((disposable) => disposable.dispose());
    });
  }
}
