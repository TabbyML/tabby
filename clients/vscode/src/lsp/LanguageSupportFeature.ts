import { commands, Uri, Position, Range } from "vscode";
import { BaseLanguageClient, StaticFeature, FeatureState, Disposable } from "vscode-languageclient";
import {
  ClientCapabilities,
  LanguageSupportDeclarationRequest,
  LanguageSupportSemanticTokensRangeRequest,
} from "tabby-agent";
import { DeclarationParams, SemanticTokensRangeParams } from "vscode-languageclient";

export class LanguageSupportFeature implements StaticFeature {
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
      languageSupport: true,
    };
  }

  preInitialize(): void {
    // nothing
  }

  initialize(): void {
    this.disposables.push(
      this.client.onRequest(LanguageSupportDeclarationRequest.type, async (params: DeclarationParams) => {
        const result = await commands.executeCommand(
          "vscode.executeDefinitionProvider",
          Uri.parse(params.textDocument.uri),
          new Position(params.position.line, params.position.character),
        );
        const items = Array.isArray(result) ? result : [result];
        const locations = items.map((item) => {
          return {
            uri: "targetUri" in item ? item.targetUri.toString() : item.uri.toString(),
            range:
              "targetRange" in item
                ? {
                    start: {
                      line: item.targetRange.start.line,
                      character: item.targetRange.start.character,
                    },
                    end: {
                      line: item.targetRange.end.line,
                      character: item.targetRange.end.character,
                    },
                  }
                : {
                    start: {
                      line: item.range.start.line,
                      character: item.range.start.character,
                    },
                    end: {
                      line: item.range.end.line,
                      character: item.range.end.character,
                    },
                  },
          };
        });
        return locations;
      }),
    );
    this.disposables.push(
      this.client.onRequest(
        LanguageSupportSemanticTokensRangeRequest.type,
        async (params: SemanticTokensRangeParams) => {
          return {
            legend: await commands.executeCommand(
              "vscode.provideDocumentRangeSemanticTokensLegend",
              Uri.parse(params.textDocument.uri),
              new Range(
                params.range.start.line,
                params.range.start.character,
                params.range.end.line,
                params.range.end.character,
              ),
            ),
            tokens: await commands.executeCommand(
              "vscode.provideDocumentRangeSemanticTokens",
              Uri.parse(params.textDocument.uri),
              new Range(
                params.range.start.line,
                params.range.start.character,
                params.range.end.line,
                params.range.end.character,
              ),
            ),
          };
        },
      ),
    );
  }

  clear(): void {
    this.disposables.forEach((disposable) => disposable.dispose());
    this.disposables = [];
  }
}
