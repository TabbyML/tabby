import type { Feature } from "../feature";
import type { TextDocumentReader, TextDocumentRangeContext } from "./documentContexts";
import { getLogger } from "../logger";
import type { Location, CancellationToken, Position } from "vscode-languageserver-protocol";
import {
  ClientCapabilities,
  LanguageSupportDeclarationRequest,
  LanguageSupportSemanticTokensRangeRequest,
  ServerCapabilities,
} from "../protocol";
import type { Connection } from "vscode-languageserver";
import { intersectionRange, isPositionInRange } from "../utils/range";

export class DeclarationSnippetsProvider implements Feature {
  private readonly logger = getLogger("DeclarationSnippetsProvider");

  private lspConnection: Connection | undefined = undefined;
  private clientCapabilities: ClientCapabilities | undefined = undefined;

  constructor(private readonly documentReader: TextDocumentReader) {}

  initialize(connection: Connection, clientCapabilities: ClientCapabilities): ServerCapabilities {
    this.lspConnection = connection;
    this.clientCapabilities = clientCapabilities;
    return {};
  }

  async collect(
    location: Location,
    limit: number | undefined,
    noReverse = false,
    token: CancellationToken,
  ): Promise<TextDocumentRangeContext[] | undefined> {
    if (!this.lspConnection || !this.clientCapabilities?.tabby?.languageSupport) {
      return undefined;
    }
    this.logger.trace("Collecting snippets for:", { location });
    const extractedSymbols = await this.extractSemanticTokenPositions(location, token);
    if (!extractedSymbols) {
      return undefined;
    }
    const allowedSymbolTypes = [
      "class",
      "decorator",
      "enum",
      "function",
      "interface",
      "macro",
      "method",
      "namespace",
      "struct",
      "type",
      "typeParameter",
    ];
    const symbols = extractedSymbols.filter((symbol) => symbol.type && allowedSymbolTypes.includes(symbol.type));
    this.logger.trace("Found symbols:", { symbols });

    // Loop through the symbol positions backwards
    const snippets: TextDocumentRangeContext[] = [];

    for (
      let symbolIndex = noReverse ? 0 : symbols.length - 1;
      noReverse ? symbolIndex < symbols.length : symbolIndex >= 0;
      noReverse ? symbolIndex++ : symbolIndex--
    ) {
      if (limit != undefined && snippets.length >= limit) {
        // Stop collecting snippets if the max number of snippets is reached
        break;
      }
      const sourcePosition = symbols[symbolIndex]?.position;
      if (!sourcePosition) {
        continue;
      }
      const result = await this.lspConnection.sendRequest(
        LanguageSupportDeclarationRequest.type,
        {
          textDocument: { uri: location.uri },
          position: sourcePosition,
        },
        token,
      );
      if (!result) {
        continue;
      }
      const firstResult = Array.isArray(result) ? result[0] : result;
      if (!firstResult) {
        continue;
      }

      const target: Location = {
        uri: "targetUri" in firstResult ? firstResult.targetUri : firstResult.uri,
        range: "targetRange" in firstResult ? firstResult.targetRange : firstResult.range,
      };
      if (target.uri == location.uri && isPositionInRange(target.range.start, location.range)) {
        // Skipping snippet as it is contained in the source location
        // this also includes the case of the symbol's declaration is at this position itself
        continue;
      }
      if (
        snippets.find(
          (snippet) => target.uri == snippet.uri && (!snippet.range || intersectionRange(target.range, snippet.range)),
        )
      ) {
        // Skipping snippet as it is already collected
        continue;
      }

      const snippet = await this.documentReader.read(target.uri, target.range, token);
      if (snippet) {
        snippets.push(snippet);
      }
    }
    this.logger.trace("Collected snippets:", snippets);
    return snippets;
  }

  private async extractSemanticTokenPositions(
    location: Location,
    token: CancellationToken,
  ): Promise<
    | {
        position: Position;
        type: string | undefined;
      }[]
    | undefined
  > {
    if (!this.lspConnection || !this.clientCapabilities?.tabby?.languageSupport) {
      return undefined;
    }

    const result = await this.lspConnection.sendRequest(
      LanguageSupportSemanticTokensRangeRequest.type,
      {
        textDocument: { uri: location.uri },
        range: location.range,
      },
      token,
    );
    if (!result || !result.legend || !result.legend.tokenTypes || !result.tokens || !result.tokens.data) {
      return undefined;
    }
    const { legend, tokens } = result;
    const data: number[] = Array.isArray(tokens.data) ? tokens.data : Object.values(tokens.data);
    const semanticSymbols: {
      position: Position;
      type: string | undefined;
    }[] = [];
    let line = 0;
    let character = 0;
    for (let i = 0; i + 4 < data.length; i += 5) {
      const deltaLine = data[i];
      const deltaChar = data[i + 1];
      // i + 2 is token length, not used here
      const typeIndex = data[i + 3];
      // i + 4 is type modifiers, not used here
      if (deltaLine === undefined || deltaChar === undefined || typeIndex === undefined) {
        break;
      }

      line += deltaLine;
      if (deltaLine > 0) {
        character = deltaChar;
      } else {
        character += deltaChar;
      }
      semanticSymbols.push({
        position: { line, character },
        type: legend.tokenTypes[typeIndex],
      });
    }
    return semanticSymbols;
  }
}
