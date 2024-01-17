import * as vscode from "vscode";

import { DocumentContext } from "../get-current-doc-context";
import { ContextSnippet } from "../types";

import { ContextStrategy, ContextStrategyFactory } from "./context-strategy";

export interface GetContextOptions {
  document: vscode.TextDocument;
  position: vscode.Position;
  docContext: DocumentContext;
  abortSignal?: AbortSignal;
  maxChars: number;
}

// k parameter for the reciprocal rank fusion scoring. 60 is the default value in many places
//
// c.f. https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking#how-rrf-ranking-works
const RRF_K = 60;

export interface ContextSummary {
  /** Name of the strategy being used */
  strategy: ContextStrategy;
  /** Total duration of the context retrieval phase */
  duration: number;
  /** Total characters of combined context snippets */
  totalChars: number;
  /** Detailed information for each retriever that has run */
  retrieverStats: {
    [identifier: string]: {
      /** Number of items that are ended up being suggested to be used by the context mixer */
      suggestedItems: number;
      /** Number of total snippets */
      retrievedItems: number;
      /** Duration of the individual retriever */
      duration: number;
      /**
       * A bitmap that indicates at which position in the result set an entry from the given
       * retriever is included. It only includes information about the first 32 entries.
       *
       * The lowest bit indicates if the first entry is included, the second lowest bit
       * indicates if the second entry is included, and so on.
       */
      positionBitmap: number;
    };
  };
}

export interface GetContextResult {
  context: ContextSnippet[];
  logSummary: ContextSummary;
}

/**
 * The context mixer is responsible for combining multiple context retrieval strategies into a
 * single proposed context list.
 *
 * This is done by ranking the order of documents using reciprocal rank fusion and then combining
 * the snippets from each retriever into a single list using top-k (so we will pick all returned
 * ranged for the top ranked document from all retrieval sources before we move on to the second
 * document).
 */
export class ContextMixer implements vscode.Disposable {
  constructor(private strategyFactory: ContextStrategyFactory) {}

  public async getContext(
    options: GetContextOptions
  ): Promise<GetContextResult> {
    const start = performance.now();

    const { name: strategy, retrievers } = this.strategyFactory.getStrategy(
      options.document
    );
    if (retrievers.length === 0) {
      return {
        context: [],
        logSummary: {
          strategy: "none",
          totalChars: 0,
          duration: 0,
          retrieverStats: {},
        },
      };
    }

    const results = await Promise.all(
      retrievers.map(async (retriever) => {
        const retrieverStart = performance.now();
        const snippets = await retriever.retrieve({
          ...options,
          hints: {
            maxChars: options.maxChars,
            maxMs: 150,
          },
        });

        return {
          identifier: retriever.identifier,
          duration: performance.now() - retrieverStart,
          snippets,
        };
      })
    );

    // For every retrieval strategy, create a map of snippets by document.
    const resultsByDocument = new Map<
      string,
      { [identifier: string]: ContextSnippet[] }
    >();
    for (const { identifier, snippets } of results) {
      for (const snippet of snippets) {
        const documentId = snippet.fileName;

        let document = resultsByDocument.get(documentId);
        if (!document) {
          document = {};
          resultsByDocument.set(documentId, document);
        }
        if (!document[identifier]) {
          document[identifier] = [];
        }
        document[identifier].push(snippet);
      }
    }

    // Rank the order of documents using reciprocal rank fusion.
    //
    // For this, we take the top rank of every document from each retrieved set and compute a
    // combined rank. The idea is that a document that ranks highly across multiple retrievers
    // should be ranked higher overall.
    const fusedDocumentScores: Map<string, number> = new Map();
    for (const { identifier, snippets } of results) {
      snippets.forEach((snippet, rank) => {
        const documentId = snippet.fileName;

        // Since every retriever can return many snippets for a given document, we need to
        // only consider the best rank for each document.
        // We can use the previous map by document to find the highest ranked snippet for a
        // retriever
        const isBestRankForRetriever =
          resultsByDocument.get(documentId)?.[identifier][0] === snippet;
        if (!isBestRankForRetriever) {
          return;
        }

        const reciprocalRank = 1 / (RRF_K + rank);

        const score = fusedDocumentScores.get(documentId);
        if (score === undefined) {
          fusedDocumentScores.set(documentId, reciprocalRank);
        } else {
          fusedDocumentScores.set(documentId, score + reciprocalRank);
        }
      });
    }

    const fusedDocuments = [...fusedDocumentScores.entries()]
      .sort((a, b) => b[1] - a[1])
      .map((e) => e[0]);

    const mixedContext: ContextSnippet[] = [];
    const retrieverStats: ContextSummary["retrieverStats"] = {};
    let totalChars = 0;
    let position = 0;
    // Now that we have a sorted list of documents (with the first document being the highest
    // ranked one), we use top-k to combine snippets from each retriever into a result set.
    //
    // We start with the highest ranked document and include all retrieved snippets from this
    // document into the result set, starting with the top retrieved snippet from each retriever
    // and adding entries greedily.
    for (const documentId of fusedDocuments) {
      const resultByDocument = resultsByDocument.get(documentId);
      if (!resultByDocument) {
        continue;
      }

      // We want to start iterating over every retrievers first rank, then every retrievers
      // second rank etc. The termination criteria is thus defined to be the length of the
      // largest snippet list of any retriever.
      const maxMatches = Math.max(
        ...Object.values(resultByDocument).map((r) => r.length)
      );

      for (let i = 0; i < maxMatches; i++) {
        for (const [identifier, snippets] of Object.entries(resultByDocument)) {
          if (i >= snippets.length) {
            continue;
          }
          const snippet = snippets[i];
          if (totalChars + snippet.content.length > options.maxChars) {
            continue;
          }

          mixedContext.push(snippet);
          totalChars += snippet.content.length;

          if (!retrieverStats[identifier]) {
            retrieverStats[identifier] = {
              suggestedItems: 0,
              positionBitmap: 0,
              retrievedItems:
                results.find((r) => r.identifier === identifier)?.snippets
                  .length ?? 0,
              duration:
                results.find((r) => r.identifier === identifier)?.duration ?? 0,
            };
          }

          retrieverStats[identifier].suggestedItems++;
          if (position < 32) {
            retrieverStats[identifier].positionBitmap |= 1 << position;
          }

          position++;
        }
      }
    }

    const logSummary: ContextSummary = {
      strategy,
      duration: performance.now() - start,
      totalChars,
      retrieverStats,
    };

    return {
      context: mixedContext,
      logSummary,
    };
  }

  public dispose(): void {
    this.strategyFactory.dispose();
  }
}
