import { commands,workspace, Uri, Position, Range ,SymbolInformation} from "vscode";
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
        //console.log(params)
        const result_name = await commands.executeCommand(
          "vscode.executeDefinitionProvider",
          Uri.parse(params.textDocument.uri),
          new Position(params.position.line, params.position.character),
        );
        const result_items = Array.isArray(result_name) ? result_name : [result_name];
        const result : SymbolInformation [] = []

        for (const item of result_items) {
          const document = await workspace.openTextDocument(Uri.parse(item.uri.path));
          const text = document.getText(item.range);
          const symbols = await commands.executeCommand<SymbolInformation[]>(
            'vscode.executeDocumentSymbolProvider',
            Uri.parse(item.uri.path)
          );
          const functionSymbol = symbols.find(symbol => symbol.name === text);
          if (functionSymbol) {
            result.push(functionSymbol);
            
          }
          
        }
        const items = Array.isArray(result) ? result : [result];
        const locations = items.map((item) => {
          return {
            uri: item.location.uri.path,
            range:{
              start: {
                line: item.location.range.start.line,
                character: item.location.range.start.character,
              },
              end: {
                line: item.location.range.end.line,
                character: item.location.range.end.character,
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
