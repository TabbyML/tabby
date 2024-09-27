import type {
  Connection,
  CancellationToken,
  Position,
  Range,
  Location,
  NotebookDocuments,
  NotebookDocument,
  NotebookCell,
  CompletionParams,
  InlineCompletionParams,
  TextDocumentPositionParams,
} from "vscode-languageserver";
import type { TextDocument } from "vscode-languageserver-textdocument";
import type { TextDocuments } from "../lsp/textDocuments";
import type { AnonymousUsageLogger } from "../telemetry";
import type { Feature } from "../feature";
import type { Configurations } from "../config";
import type { TabbyApiClient } from "../http/tabbyApiClient";
import type { GitContextProvider } from "../git";
import type { RecentlyChangedCodeSearch } from "../codeSearch/recentlyChanged";
import {
  RegistrationRequest,
  UnregistrationRequest,
  CompletionTriggerKind,
  InlineCompletionTriggerKind,
  CompletionItemKind,
} from "vscode-languageserver";
import {
  ClientCapabilities,
  ServerCapabilities,
  TextDocumentCompletionFeatureRegistration,
  TextDocumentInlineCompletionFeatureRegistration,
  CompletionList,
  CompletionItem as LspCompletionItem,
  InlineCompletionRequest,
  InlineCompletionList,
  InlineCompletionItem,
  TelemetryEventNotification,
  EventParams,
  LanguageSupportDeclarationRequest,
  LanguageSupportSemanticTokensRangeRequest,
  ReadFileRequest,
  ReadFileParams,
  EditorOptionsRequest,
  EditorOptions,
  GitRepository,
} from "../protocol";
import { CompletionCache } from "./cache";
import { CompletionDebounce } from "./debounce";
import { CompletionStats } from "./statistics";
import { CompletionContext, CompletionRequest } from "./contexts";
import { CompletionSolution, CompletionItem } from "./solution";
import { preCacheProcess, postCacheProcess } from "./postprocess";
import { getLogger } from "../logger";
import { abortSignalFromAnyOf } from "../utils/signal";
import { splitLines, extractNonReservedWordList } from "../utils/string";
import { MutexAbortError, isCanceledError } from "../utils/error";
import { isPositionInRange, intersectionRange } from "../utils/range";

export class CompletionProvider implements Feature {
  private readonly logger = getLogger("CompletionProvider");

  private readonly completionCache = new CompletionCache();
  private readonly completionDebounce = new CompletionDebounce();
  private readonly completionStats = new CompletionStats();

  private submitStatsTimer: ReturnType<typeof setInterval> | undefined = undefined;

  private lspConnection: Connection | undefined = undefined;
  private clientCapabilities: ClientCapabilities | undefined = undefined;

  private mutexAbortController: AbortController | undefined = undefined;

  constructor(
    private readonly configurations: Configurations,
    private readonly tabbyApiClient: TabbyApiClient,
    private readonly documents: TextDocuments<TextDocument>,
    private readonly notebooks: NotebookDocuments<TextDocument>,
    private readonly anonymousUsageLogger: AnonymousUsageLogger,
    private readonly gitContextProvider: GitContextProvider,
    private readonly recentlyChangedCodeSearch: RecentlyChangedCodeSearch,
  ) {}

  initialize(connection: Connection, clientCapabilities: ClientCapabilities): ServerCapabilities {
    this.lspConnection = connection;
    this.clientCapabilities = clientCapabilities;

    let serverCapabilities: ServerCapabilities = {};
    if (clientCapabilities.textDocument?.completion) {
      connection.onCompletion(async (params, token) => {
        return this.provideCompletion(params, token);
      });
      serverCapabilities = {
        ...serverCapabilities,
        completionProvider: {},
      };
    }
    if (clientCapabilities.textDocument?.inlineCompletion) {
      connection.onRequest(InlineCompletionRequest.type, async (params, token) => {
        return this.provideInlineCompletion(params, token);
      });
      serverCapabilities = {
        ...serverCapabilities,
        inlineCompletionProvider: true,
      };
    }
    connection.onNotification(TelemetryEventNotification.type, async (param) => {
      return this.postEvent(param);
    });

    const submitStatsInterval = 1000 * 60 * 60 * 24; // 24h
    this.submitStatsTimer = setInterval(async () => {
      await this.submitStats();
    }, submitStatsInterval);

    return serverCapabilities;
  }

  async initialized(connection: Connection) {
    await this.syncFeatureRegistration(connection);
    this.tabbyApiClient.on("statusUpdated", async () => {
      await this.syncFeatureRegistration(connection);
    });
  }

  private async syncFeatureRegistration(connection: Connection) {
    if (this.tabbyApiClient.isCodeCompletionApiAvailable()) {
      if (this.clientCapabilities?.textDocument?.completion) {
        connection.sendRequest(RegistrationRequest.type, {
          registrations: [
            {
              id: TextDocumentCompletionFeatureRegistration.type.method,
              method: TextDocumentCompletionFeatureRegistration.type.method,
            },
          ],
        });
      }
      if (this.clientCapabilities?.textDocument?.inlineCompletion) {
        connection.sendRequest(RegistrationRequest.type, {
          registrations: [
            {
              id: TextDocumentInlineCompletionFeatureRegistration.type.method,
              method: TextDocumentInlineCompletionFeatureRegistration.type.method,
            },
          ],
        });
      }
    } else {
      if (this.clientCapabilities?.textDocument?.completion) {
        connection.sendRequest(UnregistrationRequest.type, {
          unregisterations: [
            {
              id: TextDocumentCompletionFeatureRegistration.type.method,
              method: TextDocumentCompletionFeatureRegistration.type.method,
            },
          ],
        });
      }
      if (this.clientCapabilities?.textDocument?.inlineCompletion) {
        connection.sendRequest(UnregistrationRequest.type, {
          unregisterations: [
            {
              id: TextDocumentInlineCompletionFeatureRegistration.type.method,
              method: TextDocumentInlineCompletionFeatureRegistration.type.method,
            },
          ],
        });
      }
    }
  }

  async shutdown(): Promise<void> {
    await this.submitStats();
    if (this.submitStatsTimer) {
      clearInterval(this.submitStatsTimer);
    }
  }

  async provideCompletion(params: CompletionParams, token: CancellationToken): Promise<CompletionList | null> {
    if (!this.tabbyApiClient.isCodeCompletionApiAvailable()) {
      throw {
        name: "CodeCompletionFeatureNotAvailableError",
        message: "Code completion feature not available",
      };
    }
    if (token.isCancellationRequested) {
      return null;
    }
    const abortController = new AbortController();
    token.onCancellationRequested(() => abortController.abort());
    try {
      const request = await this.completionParamsToCompletionRequest(params, token);
      if (!request) {
        return null;
      }
      const response = await this.provideCompletions(request.request, abortController.signal);
      if (!response) {
        return null;
      }
      return this.toCompletionList(response, params, request.additionalPrefixLength);
    } catch (error) {
      return null;
    }
  }

  async provideInlineCompletion(
    params: InlineCompletionParams,
    token: CancellationToken,
  ): Promise<InlineCompletionList | null> {
    if (!this.tabbyApiClient.isCodeCompletionApiAvailable()) {
      throw {
        name: "CodeCompletionFeatureNotAvailableError",
        message: "Code completion feature not available",
      };
    }
    if (token.isCancellationRequested) {
      return null;
    }
    const abortController = new AbortController();
    token.onCancellationRequested(() => abortController.abort());
    try {
      const request = await this.inlineCompletionParamsToCompletionRequest(params, token);
      if (!request) {
        return null;
      }
      const response = await this.provideCompletions(request.request, abortController.signal);
      if (!response) {
        return null;
      }
      return this.toInlineCompletionList(response, params, request.additionalPrefixLength);
    } catch (error) {
      return null;
    }
  }

  async postEvent(params: EventParams): Promise<void> {
    this.completionStats.addEvent(params.type);
    const request = {
      type: params.type,
      select_kind: params.selectKind,
      completion_id: params.eventId.completionId,
      choice_index: params.eventId.choiceIndex,
      view_id: params.viewId,
      elapsed: params.elapsed,
    };
    await this.tabbyApiClient.postEvent(request);
  }

  private async completionParamsToCompletionRequest(
    params: CompletionParams,
    token?: CancellationToken,
  ): Promise<{ request: CompletionRequest; additionalPrefixLength?: number } | null> {
    const result = await this.textDocumentPositionParamsToCompletionRequest(params, token);
    if (!result) {
      return null;
    }
    result.request.manually = params.context?.triggerKind === CompletionTriggerKind.Invoked;
    return result;
  }

  private async inlineCompletionParamsToCompletionRequest(
    params: InlineCompletionParams,
    token?: CancellationToken,
  ): Promise<{ request: CompletionRequest; additionalPrefixLength?: number } | null> {
    const result = await this.textDocumentPositionParamsToCompletionRequest(params, token);
    if (!result) {
      return null;
    }
    result.request.manually = params.context?.triggerKind === InlineCompletionTriggerKind.Invoked;
    return result;
  }

  private toCompletionList(
    solution: CompletionSolution,
    documentPosition: TextDocumentPositionParams,
    additionalPrefixLength: number = 0,
  ): CompletionList | null {
    const { textDocument, position } = documentPosition;
    const document = this.documents.get(textDocument.uri);
    if (!document) {
      return null;
    }

    // Get word prefix if cursor is at end of a word
    const linePrefix = document.getText({
      start: { line: position.line, character: 0 },
      end: position,
    });
    const wordPrefix = linePrefix.match(/(\w+)$/)?.[0] ?? "";

    const list = solution.toInlineCompletionList();
    return {
      isIncomplete: list.isIncomplete,
      items: list.items.map((item): LspCompletionItem => {
        const insertionText = item.insertText.slice(
          document.offsetAt(position) - (item.range.start - additionalPrefixLength),
        );

        const lines = splitLines(insertionText);
        const firstLine = lines[0] || "";
        const secondLine = lines[1] || "";
        return {
          label: wordPrefix + firstLine,
          labelDetails: {
            detail: secondLine,
            description: "Tabby",
          },
          kind: CompletionItemKind.Text,
          documentation: {
            kind: "markdown",
            value: `\`\`\`\n${linePrefix + insertionText}\n\`\`\`\n ---\nSuggested by Tabby.`,
          },
          textEdit: {
            newText: wordPrefix + insertionText,
            range: {
              start: { line: position.line, character: position.character - wordPrefix.length },
              end: document.positionAt(item.range.end - additionalPrefixLength),
            },
          },
          data: item.data,
        };
      }),
    };
  }

  private toInlineCompletionList(
    solution: CompletionSolution,
    documentPosition: TextDocumentPositionParams,
    additionalPrefixLength: number = 0,
  ): InlineCompletionList | null {
    const { textDocument } = documentPosition;
    const document = this.documents.get(textDocument.uri);
    if (!document) {
      return null;
    }

    const list = solution.toInlineCompletionList();
    return {
      isIncomplete: list.isIncomplete,
      items: list.items.map((item): InlineCompletionItem => {
        return {
          insertText: item.insertText,
          range: {
            start: document.positionAt(item.range.start - additionalPrefixLength),
            end: document.positionAt(item.range.end - additionalPrefixLength),
          },
          data: item.data,
        };
      }),
    };
  }

  private async provideCompletions(
    request: CompletionRequest,
    signal?: AbortSignal,
  ): Promise<CompletionSolution | null> {
    this.logger.debug("Function providedCompletions called.");

    const config = this.configurations.getMergedConfig();

    // Mutex Control
    if (this.mutexAbortController && !this.mutexAbortController.signal.aborted) {
      this.mutexAbortController.abort(new MutexAbortError());
    }
    this.mutexAbortController = new AbortController();
    const signals = abortSignalFromAnyOf([this.mutexAbortController.signal, signal]);

    // Processing request
    const context = new CompletionContext(request);
    if (!context.isValid()) {
      // Early return if request is not valid
      return null;
    }

    let solution: CompletionSolution | undefined = undefined;
    let cachedSolution: CompletionSolution | undefined = undefined;
    if (this.completionCache.has(context.hash)) {
      cachedSolution = this.completionCache.get(context.hash);
    }

    try {
      // Resolve solution
      if (cachedSolution && (!request.manually || cachedSolution.isCompleted)) {
        // Found cached solution
        // TriggerKind is Automatic, or the solution is completed
        // Return cached solution, do not need to fetch more choices

        // Debounce before continue processing cached solution
        await this.completionDebounce.debounce(
          {
            request,
            config: config.completion.debounce,
            responseTime: 0,
          },
          signals,
        );

        solution = cachedSolution.withContext(context);
        this.logger.info("Completion cache hit.");
      } else if (!request.manually) {
        // No cached solution
        // TriggerKind is Automatic
        // We need to fetch the first choice

        // Debounce before fetching
        const averageResponseTime = this.tabbyApiClient.getCompletionRequestStats().stats().stats.averageResponseTime;
        await this.completionDebounce.debounce(
          {
            request,
            config: config.completion.debounce,
            responseTime: averageResponseTime,
          },
          signals,
        );

        solution = new CompletionSolution(context);
        // Fetch the completion
        this.logger.info(`Fetching completion...`);
        try {
          const response = await this.tabbyApiClient.fetchCompletion(
            {
              language: context.language,
              segments: context.buildSegments(config.completion.prompt),
              temperature: undefined,
            },
            signals,
            this.completionStats,
          );
          const completionItem = CompletionItem.createFromResponse(context, response);
          // postprocess: preCache
          solution.add(...(await preCacheProcess([completionItem], config.postprocess)));
        } catch (error) {
          if (isCanceledError(error)) {
            this.logger.info(`Fetching completion canceled.`);
            solution = undefined;
          }
        }
      } else {
        // No cached solution, or cached solution is not completed
        // TriggerKind is Manual
        // We need to fetch the more choices

        solution = cachedSolution?.withContext(context) ?? new CompletionSolution(context);
        this.logger.info(`Fetching more completions...`);

        try {
          let tries = 0;
          while (
            solution.items.length < config.completion.solution.maxItems &&
            tries < config.completion.solution.maxTries
          ) {
            tries++;
            const response = await this.tabbyApiClient.fetchCompletion(
              {
                language: context.language,
                segments: context.buildSegments(config.completion.prompt),
                temperature: config.completion.solution.temperature,
              },
              signals,
              this.completionStats,
            );
            const completionItem = CompletionItem.createFromResponse(context, response);
            // postprocess: preCache
            solution.add(...(await preCacheProcess([completionItem], config.postprocess)));
            if (signals.aborted) {
              throw signals.reason;
            }
          }
          // Mark the solution as completed
          solution.isCompleted = true;
        } catch (error) {
          if (isCanceledError(error)) {
            this.logger.info(`Fetching completion canceled.`);
            solution = undefined;
          }
        }
      }
      // Postprocess solution
      if (solution) {
        // Update Cache
        this.completionCache.update(solution);

        // postprocess: postCache
        solution = solution.withItems(...(await postCacheProcess(solution.items, config.postprocess)));
        if (signals.aborted) {
          throw signals.reason;
        }
      }
    } catch (error) {
      if (!isCanceledError(error)) {
        this.logger.error(`Providing completions failed.`, error);
      }
    }
    if (solution) {
      this.completionStats.addProviderStatsEntry({ triggerMode: request.manually ? "manual" : "auto" });
      this.logger.info(`Completed processing completions, choices returned: ${solution.items.length}.`);
      this.logger.trace("Completion solution:", { solution: solution.toInlineCompletionList() });
    }
    return solution ?? null;
  }

  private async textDocumentPositionParamsToCompletionRequest(
    params: TextDocumentPositionParams,
    token?: CancellationToken,
  ): Promise<{ request: CompletionRequest; additionalPrefixLength?: number } | null> {
    const { textDocument, position } = params;

    this.logger.trace("Building completion context...", { uri: textDocument.uri });

    const document = this.documents.get(textDocument.uri);
    if (!document) {
      this.logger.trace("Document not found, cancelled.");
      return null;
    }

    const request: CompletionRequest = {
      filepath: document.uri,
      language: document.languageId,
      text: document.getText(),
      position: document.offsetAt(position),
    };

    const notebookCell = this.notebooks.getNotebookCell(textDocument.uri);
    let additionalContext: { prefix: string; suffix: string } | undefined = undefined;
    if (notebookCell) {
      this.logger.trace("Notebook cell found:", { cell: notebookCell.kind });
      additionalContext = this.buildNotebookAdditionalContext(document, notebookCell);
    }
    if (additionalContext) {
      this.logger.trace("Applying notebook additional context...", { additionalContext });
      request.text = additionalContext.prefix + request.text + additionalContext.suffix;
      request.position += additionalContext.prefix.length;
    }

    const connection = this.lspConnection;
    if (connection && this.clientCapabilities?.tabby?.editorOptions) {
      this.logger.trace("Collecting editor options...");
      const editorOptions: EditorOptions | null = await connection.sendRequest(
        EditorOptionsRequest.type,
        {
          uri: params.textDocument.uri,
        },
        token,
      );
      this.logger.trace("Collected editor options:", { editorOptions });
      request.indentation = editorOptions?.indentation;
    }
    if (connection && this.clientCapabilities?.workspace) {
      this.logger.trace("Collecting workspace folders...");
      const workspaceFolders = await connection.workspace.getWorkspaceFolders();
      this.logger.trace("Collected workspace folders:", { workspaceFolders });
      request.workspace = workspaceFolders?.find((folder) => document.uri.startsWith(folder.uri))?.uri;
    }
    this.logger.trace("Collecting git context...");
    const repo: GitRepository | null = await this.gitContextProvider.getRepository({ uri: document.uri }, token);
    this.logger.trace("Collected git context:", { repo });
    if (repo) {
      request.git = {
        root: repo.root,
        remotes: repo.remoteUrl ? [{ name: "", url: repo.remoteUrl }] : repo.remotes ?? [],
      };
    }
    if (connection && this.clientCapabilities?.tabby?.languageSupport) {
      request.declarations = await this.collectDeclarationSnippets(connection, document, position, token);
    }
    request.relevantSnippetsFromChangedFiles = await this.collectSnippetsFromRecentlyChangedFiles(document, position);

    this.logger.trace("Completed completion context:", { request });
    return { request, additionalPrefixLength: additionalContext?.prefix.length };
  }

  private buildNotebookAdditionalContext(
    textDocument: TextDocument,
    notebookCell: NotebookCell,
  ): { prefix: string; suffix: string } | undefined {
    this.logger.trace("Building notebook additional context...");
    const notebook = this.notebooks.findNotebookDocumentForCell(notebookCell);
    if (!notebook) {
      return notebook;
    }
    const index = notebook.cells.indexOf(notebookCell);
    const prefix = this.buildNotebookContext(notebook, 0, index, textDocument.languageId) + "\n\n";
    const suffix =
      "\n\n" + this.buildNotebookContext(notebook, index + 1, notebook.cells.length, textDocument.languageId);

    this.logger.trace("Notebook additional context:", { prefix, suffix });
    return { prefix, suffix };
  }

  private notebookLanguageComments: { [languageId: string]: (code: string) => string } = {
    markdown: (code) => "```\n" + code + "\n```",
    python: (code) =>
      code
        .split("\n")
        .map((l) => "# " + l)
        .join("\n"),
  };

  private buildNotebookContext(notebook: NotebookDocument, from: number, to: number, languageId: string): string {
    return notebook.cells
      .slice(from, to)
      .map((cell) => {
        const textDocument = this.notebooks.getCellTextDocument(cell);
        if (!textDocument) {
          return "";
        }
        if (textDocument.languageId === languageId) {
          return textDocument.getText();
        } else if (Object.keys(this.notebookLanguageComments).includes(languageId)) {
          return this.notebookLanguageComments[languageId]?.(textDocument.getText()) ?? "";
        } else {
          return "";
        }
      })
      .join("\n\n");
  }

  private async collectDeclarationSnippets(
    connection: Connection,
    textDocument: TextDocument,
    position: Position,
    token?: CancellationToken,
  ): Promise<{ filepath: string; text: string; offset?: number }[] | undefined> {
    const config = this.configurations.getMergedConfig();
    if (!config.completion.prompt.fillDeclarations.enabled) {
      return;
    }
    this.logger.debug("Collecting declaration snippets...");
    this.logger.trace("Collecting snippets for:", { textDocument: textDocument.uri, position });
    // Find symbol positions in the previous lines
    const prefixRange: Range = {
      start: { line: Math.max(0, position.line - config.completion.prompt.maxPrefixLines), character: 0 },
      end: { line: position.line, character: position.character },
    };
    const extractedSymbols = await this.extractSemanticTokenPositions(
      connection,
      {
        uri: textDocument.uri,
        range: prefixRange,
      },
      token,
    );
    if (!extractedSymbols) {
      // FIXME: fallback to simple split words positions
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
    const symbols = extractedSymbols.filter((symbol) => allowedSymbolTypes.includes(symbol.type ?? ""));
    this.logger.trace("Found symbols in prefix text:", { symbols });

    // Loop through the symbol positions backwards
    const snippets: { filepath: string; text: string; offset?: number }[] = [];
    const snippetLocations: Location[] = [];
    for (let symbolIndex = symbols.length - 1; symbolIndex >= 0; symbolIndex--) {
      if (snippets.length >= config.completion.prompt.fillDeclarations.maxSnippets) {
        // Stop collecting snippets if the max number of snippets is reached
        break;
      }
      const symbolPosition = symbols[symbolIndex]?.position;
      if (!symbolPosition) {
        continue;
      }
      const result = await connection.sendRequest(
        LanguageSupportDeclarationRequest.type,
        {
          textDocument: { uri: textDocument.uri },
          position: symbolPosition,
        },
        token,
      );
      if (!result) {
        continue;
      }
      const item = Array.isArray(result) ? result[0] : result;
      if (!item) {
        continue;
      }
      const location: Location = {
        uri: "targetUri" in item ? item.targetUri : item.uri,
        range: "targetRange" in item ? item.targetRange : item.range,
      };
      this.logger.trace("Processing declaration location...", { location });
      if (location.uri == textDocument.uri && isPositionInRange(location.range.start, prefixRange)) {
        // this symbol's declaration is already contained in the prefix range
        // this also includes the case of the symbol's declaration is at this position itself
        this.logger.trace("Skipping snippet as it is contained in the prefix.");
        continue;
      }
      if (
        snippetLocations.find(
          (collectedLocation) =>
            location.uri == collectedLocation.uri && intersectionRange(location.range, collectedLocation.range),
        )
      ) {
        this.logger.trace("Skipping snippet as it is already collected.");
        continue;
      }
      this.logger.trace("Prepare to fetch text content...");
      let text: string | undefined = undefined;
      const targetDocument = this.documents.get(location.uri);
      if (targetDocument) {
        this.logger.trace("Fetching text content from synced text document.", {
          uri: targetDocument.uri,
          range: location.range,
        });
        text = targetDocument.getText(location.range);
        this.logger.trace("Fetched text content from synced text document.", { text });
      } else if (this.clientCapabilities?.tabby?.workspaceFileSystem) {
        const params: ReadFileParams = {
          uri: location.uri,
          format: "text",
          range: {
            start: { line: location.range.start.line, character: 0 },
            end: { line: location.range.end.line, character: location.range.end.character },
          },
        };
        this.logger.trace("Fetching text content from ReadFileRequest.", { params });
        const result = await connection.sendRequest(ReadFileRequest.type, params, token);
        this.logger.trace("Fetched text content from ReadFileRequest.", { result });
        text = result?.text;
      } else {
        // FIXME: fallback to fs
      }
      if (!text) {
        this.logger.trace("Cannot fetch text content, continue to next.", { result });
        continue;
      }
      const maxChars = config.completion.prompt.fillDeclarations.maxCharsPerSnippet;
      if (text.length > maxChars) {
        // crop the text to fit within the chars limit
        text = text.slice(0, maxChars);
        const lastNewLine = text.lastIndexOf("\n");
        if (lastNewLine > 0) {
          text = text.slice(0, lastNewLine + 1);
        }
      }
      if (text.length > 0) {
        this.logger.trace("Collected declaration snippet:", { text });
        snippets.push({ filepath: location.uri, offset: targetDocument?.offsetAt(position), text });
        snippetLocations.push(location);
      }
    }
    this.logger.debug("Completed collecting declaration snippets.");
    this.logger.trace("Collected snippets:", snippets);
    return snippets;
  }

  private async extractSemanticTokenPositions(
    connection: Connection,
    location: Location,
    token?: CancellationToken,
  ): Promise<
    | {
        position: Position;
        type: string | undefined;
      }[]
    | undefined
  > {
    const result = await connection.sendRequest(
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

  private async collectSnippetsFromRecentlyChangedFiles(
    textDocument: TextDocument,
    position: Position,
  ): Promise<{ filepath: string; offset: number; text: string; score: number }[] | undefined> {
    const config = this.configurations.getMergedConfig();
    if (!config.completion.prompt.collectSnippetsFromRecentChangedFiles.enabled) {
      return undefined;
    }
    this.logger.debug("Collecting snippets from recently changed files...");
    this.logger.trace("Collecting snippets for:", { document: textDocument.uri, position });
    const prefixRange: Range = {
      start: { line: Math.max(0, position.line - config.completion.prompt.maxPrefixLines), character: 0 },
      end: { line: position.line, character: position.character },
    };
    const prefixText = textDocument.getText(prefixRange);
    const query = extractNonReservedWordList(prefixText);
    const snippets = await this.recentlyChangedCodeSearch.collectRelevantSnippets(
      query,
      textDocument,
      config.completion.prompt.collectSnippetsFromRecentChangedFiles.maxSnippets,
    );
    this.logger.debug("Completed collecting snippets from recently changed files.");
    this.logger.trace("Collected snippets:", snippets);
    return snippets;
  }

  private async submitStats() {
    const stats = this.completionStats.stats();
    if (stats["completion_request"]["count"] > 0) {
      await this.anonymousUsageLogger.event("AgentStats", { stats });
      this.completionStats.reset();
    }
  }
}
