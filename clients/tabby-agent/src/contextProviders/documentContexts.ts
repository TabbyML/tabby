import type { CancellationToken, Connection, Range } from "vscode-languageserver";
import { TextDocument } from "vscode-languageserver-textdocument";
import fs from "fs-extra";
import type { TextDocuments } from "../extensions/textDocuments";
import type { Feature } from "../feature";
import { ClientCapabilities, ServerCapabilities, ReadFileRequest, ReadFileParams } from "../protocol";
import { getLogger } from "../logger";
import { isBrowser } from "../env";
import { getLanguageId } from "../utils/languageId";

export interface TextDocumentRangeContext {
  uri: string;
  language: string;
  /**
   * If the range is provided, this context presents a range of the text document.
   */
  range?: Range;
  /**
   * The full text of the document if the range is not provided.
   * The text in the range of the document if the range is provided.
   */
  text: string;
}

export class TextDocumentReader implements Feature {
  private readonly logger = getLogger("TextDocumentReader");

  private lspConnection: Connection | undefined = undefined;
  private clientCapabilities: ClientCapabilities | undefined = undefined;

  constructor(private readonly documents: TextDocuments<TextDocument>) {}

  initialize(connection: Connection, clientCapabilities: ClientCapabilities): ServerCapabilities {
    this.lspConnection = connection;
    this.clientCapabilities = clientCapabilities;
    return {};
  }

  async read(
    documentOrUri: TextDocument | string,
    range: Range | undefined,
    token: CancellationToken | undefined,
  ): Promise<TextDocumentRangeContext | undefined> {
    let targetDocument: TextDocument | undefined = undefined;
    let targetUri: string | undefined = undefined;
    if (typeof documentOrUri === "string") {
      targetDocument = this.documents.get(documentOrUri);
      targetUri = documentOrUri;
    } else {
      targetDocument = documentOrUri;
      targetUri = documentOrUri.uri;
    }

    let context: TextDocumentRangeContext | undefined = undefined;
    if (targetDocument) {
      context = {
        uri: targetDocument.uri,
        language: targetDocument.languageId,
        range: range,
        text: targetDocument.getText(range),
      };
      this.logger.trace("Read context from synced text document.", context);
    } else if (targetUri) {
      const language = getLanguageId(targetUri);
      // read from lsp connection
      if (this.lspConnection && this.clientCapabilities?.tabby?.workspaceFileSystem) {
        try {
          const params: ReadFileParams = {
            uri: targetUri,
            format: "text",
            range: range,
          };
          const result = await this.lspConnection.sendRequest(ReadFileRequest.type, params, token);
          if (result && typeof result.text === "string") {
            context = {
              uri: targetUri,
              language: language,
              range: range,
              text: result.text,
            };
            this.logger.trace("Read context from LSP ReadFileRequest.", { result });
          }
        } catch (error) {
          this.logger.trace("ReadFileRequest failed.", { error });
        }
      }

      // fallback to fs
      if (!context && !isBrowser) {
        try {
          const fileContent = await new Promise<string>((resolve, reject) => {
            const readFilePromise = fs.readFile(targetUri, "utf-8");
            if (token) {
              token.onCancellationRequested(() => {
                reject(new Error("Operation canceled"));
              });
            }
            readFilePromise.then(resolve).catch(reject);
          });

          const textDocument = TextDocument.create(targetUri, language, 0, fileContent);
          const text = textDocument.getText(range);
          context = {
            uri: targetUri,
            language: language,
            range: range,
            text: text,
          };
          this.logger.trace("Read context from file system.", { text });
        } catch (error) {
          this.logger.trace("Read file failed.", { error });
        }
      }
    }
    return context;
  }
}
