import { EventEmitter } from "events";
import type {
  Connection,
  CancellationToken,
  Disposable,
  NotebookDocuments,
  CompletionParams,
  CompletionOptions,
  InlineCompletionParams,
  TextDocumentPositionParams,
  SelectedCompletionInfo,
} from "vscode-languageserver";
import {
  CompletionRequest as LspCompletionRequest,
  CompletionTriggerKind,
  InlineCompletionTriggerKind,
} from "vscode-languageserver";
import type { TextDocument } from "vscode-languageserver-textdocument";
import type { TextDocuments } from "../extensions/textDocuments";
import {
  ClientCapabilities,
  ServerCapabilities,
  CompletionList,
  InlineCompletionRequest,
  InlineCompletionList,
  TelemetryEventNotification,
  EventParams,
} from "../protocol";
import type { Feature } from "../feature";
import type { Configurations } from "../config";
import type { ConfigData } from "../config/type";
import type { TabbyApiClient, TabbyApiClientStatus } from "../http/tabbyApiClient";
import type { TextDocumentReader } from "../contextProviders/documentContexts";
import type { WorkspaceContextProvider } from "../contextProviders/workspace";
import type { GitContextProvider } from "../contextProviders/git";
import type { DeclarationSnippetsProvider } from "../contextProviders/declarationSnippets";
import type { RecentlyChangedCodeSearch } from "../contextProviders/recentlyChangedCodeSearch";
import type { EditorVisibleRangesTracker } from "../contextProviders/editorVisibleRanges";
import type { EditorOptionsProvider } from "../contextProviders/editorOptions";
import type { AnonymousUsageLogger } from "../telemetry";
import { calculateCompletionContextHash, CompletionCache, generateForwardingContexts } from "./cache";
import { CompletionDebouncer, DebouncingContext } from "./debouncer";
import { CompletionStatisticsEntry, CompletionStatisticsTracker } from "./statistics";
import { buildCompletionContext, CompletionContext } from "./contexts";
import { CompletionSolution, createCompletionResultItemFromResponse } from "./solution";
import { extractNonReservedWordList } from "../utils/string";
import { MutexAbortError, formatErrorMessage, isCanceledError, isRateLimitExceededError } from "../utils/error";
import { preCacheProcess, postCacheProcess } from "./postprocess";
import { buildRequest } from "./buildRequest";
import { analyzeMetrics, buildHelpMessageForLatencyIssue, LatencyTracker } from "./latencyTracker";
import { rangeInDocument } from "../utils/range";
import { getLogger } from "../logger";

export class CompletionProvider extends EventEmitter implements Feature {
  private readonly logger = getLogger("CompletionProvider");

  private readonly cache = new CompletionCache();
  private readonly debouncer = new CompletionDebouncer();
  private readonly statisticTracker = new CompletionStatisticsTracker();
  private readonly latencyTracker = new LatencyTracker();

  private isApiAvailable = false;
  private latencyIssue: "highTimeoutRate" | "slowResponseTime" | undefined = undefined;
  private rateLimitExceeded: boolean = false;
  private fetchingCompletion: boolean = false;

  private clientCapabilities: ClientCapabilities | undefined = undefined;

  private completionFeatureOptions: CompletionOptions | undefined = undefined;
  private completionFeatureRegistration: Disposable | undefined = undefined;
  private inlineCompletionFeatureRegistration: Disposable | undefined = undefined;

  private mutexAbortController: AbortController | undefined = undefined;
  private submitStatsTimer: ReturnType<typeof setInterval> | undefined = undefined;

  constructor(
    private readonly configurations: Configurations,
    private readonly tabbyApiClient: TabbyApiClient,
    private readonly documents: TextDocuments<TextDocument>,
    private readonly notebooks: NotebookDocuments<TextDocument>,
    private readonly anonymousUsageLogger: AnonymousUsageLogger,
    private readonly textDocumentReader: TextDocumentReader,
    private readonly workspaceContextProvider: WorkspaceContextProvider,
    private readonly gitContextProvider: GitContextProvider,
    private readonly declarationSnippetsProvider: DeclarationSnippetsProvider,
    private readonly recentlyChangedCodeSearch: RecentlyChangedCodeSearch,
    private readonly editorVisibleRangesTracker: EditorVisibleRangesTracker,
    private readonly editorOptionsProvider: EditorOptionsProvider,
  ) {
    super();
  }

  isAvailable(): boolean {
    return this.isApiAvailable;
  }

  getLatencyIssue(): "highTimeoutRate" | "slowResponseTime" | undefined {
    return this.latencyIssue;
  }

  getHelpMessage(format?: "plaintext" | "markdown" | "html"): string | undefined {
    if (!this.isApiAvailable) {
      return "There is no code completion model available. Please check your server configuration.";
    }
    if (this.rateLimitExceeded) {
      return "The rate limit for the code completion API has been reached. Please try again later.";
    }
    if (this.latencyIssue) {
      return buildHelpMessageForLatencyIssue(
        this.latencyIssue,
        {
          latencyStatistics: this.latencyTracker.calculateLatencyStatistics(),
          endpoint: this.configurations.getMergedConfig().server.endpoint,
          serverHealth: this.tabbyApiClient.getServerHealth(),
        },
        format,
      );
    }
    return undefined;
  }

  isRateLimitExceeded(): boolean {
    return this.rateLimitExceeded;
  }

  isFetching(): boolean {
    return this.fetchingCompletion;
  }

  private updateIsAvailable() {
    const health = this.tabbyApiClient.getServerHealth();
    const isAvailable = !!(health && health["model"]);
    if (this.isApiAvailable != isAvailable) {
      this.isApiAvailable = isAvailable;
      this.emit("isAvailableUpdated", isAvailable);
    }
  }

  private updateLatencyIssue(issue: "highTimeoutRate" | "slowResponseTime" | undefined) {
    if (this.latencyIssue != issue) {
      this.latencyIssue = issue;
      if (issue) {
        this.logger.info(`Completion latency issue detected: ${issue}.`);
      }
      this.emit("latencyIssueUpdated", issue);
    }
  }

  private updateIsRateLimitExceeded(value: boolean) {
    if (this.rateLimitExceeded != value) {
      if (value) {
        this.logger.info(`Rate limit exceeded.`);
      }
      this.rateLimitExceeded = value;
      this.emit("isRateLimitExceededUpdated", value);
    }
  }

  private updateIsFetching(value: boolean) {
    if (this.fetchingCompletion != value) {
      this.fetchingCompletion = value;
      this.emit("isFetchingUpdated", value);
    }
  }

  initialize(connection: Connection, clientCapabilities: ClientCapabilities): ServerCapabilities {
    this.clientCapabilities = clientCapabilities;

    let serverCapabilities: ServerCapabilities = {};
    if (clientCapabilities.textDocument?.completion) {
      connection.onCompletion(async (params, token) => {
        return this.provideCompletion(params, token);
      });
      this.completionFeatureOptions = {
        resolveProvider: false,
        completionItem: {
          labelDetailsSupport: true,
        },
      };
      if (!clientCapabilities.textDocument?.completion.dynamicRegistration) {
        serverCapabilities = {
          ...serverCapabilities,
          completionProvider: this.completionFeatureOptions,
        };
      }
    }
    if (clientCapabilities.textDocument?.inlineCompletion) {
      connection.onRequest(InlineCompletionRequest.type, async (params, token) => {
        return this.provideInlineCompletion(params, token);
      });
      if (!clientCapabilities.textDocument?.inlineCompletion.dynamicRegistration) {
        serverCapabilities = {
          ...serverCapabilities,
          inlineCompletionProvider: true,
        };
      }
    }
    connection.onNotification(TelemetryEventNotification.type, async (param) => {
      return this.postEvent(param);
    });

    const config = this.configurations.getMergedConfig();
    this.debouncer.updateConfig(config.completion.debounce);
    this.configurations.on("updated", (config: ConfigData) => {
      this.debouncer.updateConfig(config.completion.debounce);
    });

    this.updateIsAvailable();
    this.tabbyApiClient.on("statusUpdated", async (status: TabbyApiClientStatus) => {
      if (status === "noConnection") {
        this.updateLatencyIssue(undefined);
        this.latencyTracker.reset();
      }

      this.updateIsAvailable();
      await this.syncFeatureRegistration(connection);
    });

    const submitStatsInterval = 1000 * 60 * 60 * 24; // 24h
    this.submitStatsTimer = setInterval(async () => {
      await this.sendCompletionStatistics();
    }, submitStatsInterval);

    return serverCapabilities;
  }

  async initialized(connection: Connection) {
    await this.syncFeatureRegistration(connection);
  }

  private async syncFeatureRegistration(connection: Connection) {
    if (this.isApiAvailable) {
      if (
        this.clientCapabilities?.textDocument?.completion?.dynamicRegistration &&
        !this.completionFeatureRegistration
      ) {
        this.completionFeatureRegistration = await connection.client.register(
          LspCompletionRequest.type,
          this.completionFeatureOptions,
        );
      }
      if (
        this.clientCapabilities?.textDocument?.inlineCompletion?.dynamicRegistration &&
        !this.inlineCompletionFeatureRegistration
      ) {
        this.inlineCompletionFeatureRegistration = await connection.client.register(InlineCompletionRequest.type);
      }
    } else {
      this.completionFeatureRegistration?.dispose();
      this.completionFeatureRegistration = undefined;
      this.inlineCompletionFeatureRegistration?.dispose();
      this.inlineCompletionFeatureRegistration = undefined;
    }
  }

  async shutdown(): Promise<void> {
    await this.sendCompletionStatistics();
    if (this.submitStatsTimer) {
      clearInterval(this.submitStatsTimer);
    }
  }

  async provideCompletion(params: CompletionParams, token: CancellationToken): Promise<CompletionList | null> {
    if (!this.isApiAvailable) {
      throw {
        name: "CodeCompletionFeatureNotAvailableError",
        message: "Code completion feature not available",
      };
    }
    if (token.isCancellationRequested) {
      return null;
    }
    try {
      const result = await this.generateCompletions(
        params,
        params.context?.triggerKind !== CompletionTriggerKind.TriggerCharacter,
        undefined,
        token,
      );
      if (!result) {
        return null;
      }
      const list = result.solution.toCompletionList(result.context);
      this.logger.info(`Provided completion items: ${list.items.length}`);
      return list;
    } catch (error) {
      return null;
    }
  }

  async provideInlineCompletion(
    params: InlineCompletionParams,
    token: CancellationToken,
  ): Promise<InlineCompletionList | null> {
    if (!this.isApiAvailable) {
      throw {
        name: "CodeCompletionFeatureNotAvailableError",
        message: "Code completion feature not available",
      };
    }
    if (token.isCancellationRequested) {
      return null;
    }
    try {
      const result = await this.generateCompletions(
        params,
        params.context?.triggerKind === InlineCompletionTriggerKind.Invoked,
        params.context?.selectedCompletionInfo,
        token,
      );
      if (!result) {
        return null;
      }
      const list = result.solution.toInlineCompletionList(result.context);
      this.logger.info(`Provided inline completion items: ${list.items.length}`);
      return list;
    } catch (error) {
      return null;
    }
  }

  async postEvent(params: EventParams): Promise<void> {
    try {
      this.statisticTracker.addEvent(params.type);
    } catch (error) {
      // ignore
    }

    try {
      const request = {
        type: params.type,
        select_kind: params.selectKind,
        completion_id: params.eventId.completionId,
        choice_index: params.eventId.choiceIndex,
        view_id: params.viewId,
        elapsed: params.elapsed,
      };
      await this.tabbyApiClient.postEvent(request);
    } catch (error) {
      // ignore
    }
  }

  private async fetchExtraContext(
    context: CompletionContext,
    solution: CompletionSolution,
    timeout: number | undefined,
    token: CancellationToken,
  ): Promise<void> {
    const config = this.configurations.getMergedConfig().completion.prompt;
    const { document, position } = context;
    const prefixRange = rangeInDocument(
      { start: { line: position.line - config.maxPrefixLines, character: 0 }, end: position },
      document,
    );

    const fetchWorkspaceContext = async () => {
      try {
        solution.extraContext.workspace = await this.workspaceContextProvider.getWorkspaceContext(document.uri);
      } catch (error) {
        this.logger.debug(`Failed to fetch workspace context: ${formatErrorMessage(error)}`);
      }
    };
    const fetchGitContext = async () => {
      try {
        solution.extraContext.git = (await this.gitContextProvider.getContext(document.uri, token)) ?? undefined;
      } catch (error) {
        this.logger.debug(`Failed to fetch git context: ${formatErrorMessage(error)}`);
      }
    };
    const fetchDeclarations = async () => {
      if (config.fillDeclarations.enabled && prefixRange) {
        this.logger.debug("Collecting declarations...");
        try {
          solution.extraContext.declarations = await this.declarationSnippetsProvider.collect(
            {
              uri: document.uri,
              range: prefixRange,
            },
            config.fillDeclarations.maxSnippets,
            false,
            token,
          );
          this.logger.debug("Completed collecting declarations.");
        } catch (error) {
          this.logger.debug(`Failed to collect declarations: ${formatErrorMessage(error)}`);
        }
      }
    };
    const fetchRecentlyChangedCodeSearchResult = async () => {
      if (config.collectSnippetsFromRecentChangedFiles.enabled && prefixRange) {
        this.logger.debug("Searching recently changed code...");
        try {
          const prefixText = document.getText(prefixRange);
          const query = extractNonReservedWordList(prefixText);
          solution.extraContext.recentlyChangedCodeSearchResult = await this.recentlyChangedCodeSearch.search(
            query,
            [document.uri],
            document.languageId,
            config.collectSnippetsFromRecentChangedFiles.maxSnippets,
          );
          this.logger.debug("Completed searching recently changed code.");
        } catch (error) {
          this.logger.debug(`Failed to do recently changed code search: ${formatErrorMessage(error)}`);
        }
      }
    };
    const fetchLastViewedSnippets = async () => {
      if (config.collectSnippetsFromRecentOpenedFiles.enabled) {
        try {
          const ranges = await this.editorVisibleRangesTracker.getHistoryRanges({
            max: config.collectSnippetsFromRecentOpenedFiles.maxOpenedFiles,
            excludedUris: [document.uri],
          });
          solution.extraContext.lastViewedSnippets = (
            await ranges?.mapAsync(async (range) => {
              return await this.textDocumentReader.read(range.uri, range.range, token);
            })
          )?.filter((item) => item !== undefined);
        } catch (error) {
          this.logger.debug(`Failed to read last viewed snippets: ${formatErrorMessage(error)}`);
        }
      }
    };
    const fetchEditorOptions = async () => {
      try {
        solution.extraContext.editorOptions = await this.editorOptionsProvider.getEditorOptions(document.uri, token);
      } catch (error) {
        this.logger.debug(`Failed to fetch editor options: ${formatErrorMessage(error)}`);
      }
    };

    await new Promise<void>((resolve, reject) => {
      const disposables: Disposable[] = [];
      const disposeAll = () => {
        disposables.forEach((d) => d.dispose());
      };

      Promise.all([
        fetchWorkspaceContext(),
        fetchGitContext(),
        fetchDeclarations(),
        fetchRecentlyChangedCodeSearchResult(),
        fetchLastViewedSnippets(),
        fetchEditorOptions(),
      ]).then(() => {
        disposeAll();
        resolve();
      });
      // No need to catch Promise.all errors here, as individual fetches handle their errors.

      if (token) {
        if (token.isCancellationRequested) {
          disposeAll();
          reject(new Error("Request canceled."));
        }
        disposables.push(
          token.onCancellationRequested(() => {
            disposeAll();
            reject(new Error("Request canceled."));
          }),
        );
      }
      if (timeout) {
        const timer = setTimeout(() => {
          disposeAll();
          reject(new Error("Timeout."));
        }, timeout);
        disposables.push({
          dispose: () => {
            clearTimeout(timer);
          },
        });
      }
    });
  }

  private async generateCompletions(
    documentPosition: TextDocumentPositionParams,
    manuallyTriggered: boolean,
    selectedCompletionInfo: SelectedCompletionInfo | undefined,
    token: CancellationToken,
  ): Promise<{ context: CompletionContext; solution: CompletionSolution } | null> {
    this.logger.info("Generating completions...");
    const config = this.configurations.getMergedConfig();

    // Mutex Control
    if (this.mutexAbortController && !this.mutexAbortController.signal.aborted) {
      this.mutexAbortController.abort(new MutexAbortError());
    }
    const abortController = new AbortController();
    if (token) {
      token.onCancellationRequested(() => abortController.abort());
    }
    this.mutexAbortController = abortController;
    const signal = abortController.signal;

    // Build the context
    const { textDocument, position } = documentPosition;

    this.logger.trace("Building completion context...", { uri: textDocument.uri });

    const document = this.documents.get(textDocument.uri);
    if (!document) {
      this.logger.debug("Document not found, cancelled.");
      return null;
    }

    let notebookCells: TextDocument[] | undefined = undefined;
    const notebookCell = this.notebooks.getNotebookCell(textDocument.uri);
    if (notebookCell) {
      const notebook = this.notebooks.findNotebookDocumentForCell(notebookCell);
      if (notebook) {
        this.logger.trace("Notebook found:", { notebook: notebook.uri, cell: notebookCell.document });
        notebookCells = notebook.cells
          .map((cell) => this.notebooks.getCellTextDocument(cell))
          .filter((item) => item !== undefined);
      }
    }

    const context = buildCompletionContext(document, position, selectedCompletionInfo, notebookCells);
    this.logger.trace("Completed Building completion context.");
    const hash = calculateCompletionContextHash(context, this.documents);
    this.logger.trace("Completion hash: ", { hash });

    let solution: CompletionSolution | undefined = undefined;
    if (this.cache.has(hash)) {
      solution = this.cache.get(hash);
    }

    const debouncingContext: DebouncingContext = {
      triggerCharacter: context.currentLinePrefix.slice(-1),
      isLineEnd: context.isLineEnd,
      isDocumentEnd: !!context.suffix.match(/^\W*$/),
      manually: manuallyTriggered,
    };

    const latencyStatsList: CompletionStatisticsEntry[] = [];

    try {
      // Resolve solution
      if (solution && (!manuallyTriggered || solution.isCompleted)) {
        // Found cached solution
        // TriggerKind is Automatic, or the solution is completed
        // Return cached solution, do not need to fetch more choices

        // Debounce before continue processing cached solution
        await this.debouncer.debounce(debouncingContext, signal);
        this.logger.info("Completion cache hit.");
      } else if (!manuallyTriggered) {
        // No cached solution
        // TriggerKind is Automatic
        // We need to fetch the first choice

        solution = new CompletionSolution();

        // Debounce before fetching
        const averageResponseTime = this.latencyTracker.calculateLatencyStatistics().metrics.averageResponseTime;
        await this.debouncer.debounce(
          {
            ...debouncingContext,
            estimatedResponseTime: averageResponseTime,
          },
          signal,
        );

        try {
          const extraContextTimeout = 500; // 500ms when automatic trigger
          this.logger.info(`Fetching extra completion context with ${extraContextTimeout}ms timeout ...`);
          await this.fetchExtraContext(context, solution, extraContextTimeout, token);
        } catch (error) {
          this.logger.info(`Failed to fetch extra context: ${formatErrorMessage(error)}`);
        }
        if (signal.aborted) {
          throw signal.reason;
        }

        // Fetch the completion
        this.logger.info(`Fetching completions from the server...`);
        this.updateIsFetching(true);
        try {
          const latencyStats: CompletionStatisticsEntry = {};
          latencyStatsList.push(latencyStats);
          const response = await this.tabbyApiClient.fetchCompletion(
            {
              language: context.document.languageId,
              segments: buildRequest({
                context: context,
                extraContexts: solution.extraContext,
                config: config.completion.prompt,
              }),
              temperature: undefined,
            },
            signal,
            latencyStats,
          );
          this.updateIsRateLimitExceeded(false);

          const completionResultItem = createCompletionResultItemFromResponse(response);
          // postprocess: preCache
          const postprocessed = await preCacheProcess(
            [completionResultItem],
            context,
            solution.extraContext,
            config.postprocess,
          );
          solution.items.push(...postprocessed);
        } catch (error) {
          if (isCanceledError(error)) {
            this.logger.info(`Fetching completion canceled.`);
            solution = undefined;
          } else if (isRateLimitExceededError(error)) {
            this.updateIsRateLimitExceeded(true);
          } else {
            this.updateIsRateLimitExceeded(false);
          }
        }
      } else {
        // No cached solution, or cached solution is not completed
        // TriggerKind is Manual
        // We need to fetch the more choices

        solution = solution ?? new CompletionSolution();

        // Fetch multiple times to get more choices
        this.logger.info(`Fetching more completions from the server...`);
        this.updateIsFetching(true);

        try {
          this.logger.info(`Fetching extra completion context...`);
          await this.fetchExtraContext(context, solution, undefined, token);
        } catch (error) {
          this.logger.info(`Failed to fetch extra context: ${formatErrorMessage(error)}`);
        }
        if (signal.aborted) {
          throw signal.reason;
        }

        try {
          let tries = 0;
          while (
            solution.items.length < config.completion.solution.maxItems &&
            tries < config.completion.solution.maxTries
          ) {
            tries++;
            const latencyStats: CompletionStatisticsEntry = {};
            latencyStatsList.push(latencyStats);
            const response = await this.tabbyApiClient.fetchCompletion(
              {
                language: context.document.languageId,
                segments: buildRequest({
                  context: context,
                  extraContexts: solution.extraContext,
                  config: config.completion.prompt,
                }),
                temperature: config.completion.solution.temperature,
              },
              signal,
              latencyStats,
            );
            this.updateIsRateLimitExceeded(false);

            const completionResultItem = createCompletionResultItemFromResponse(response);
            // postprocess: preCache
            const postprocessed = await preCacheProcess(
              [completionResultItem],
              context,
              solution.extraContext,
              config.postprocess,
            );
            solution.items.push(...postprocessed);
            if (signal.aborted) {
              throw signal.reason;
            }
          }
          // Mark the solution as completed
          solution.isCompleted = true;
        } catch (error) {
          if (isCanceledError(error)) {
            this.logger.info(`Fetching completion canceled.`);
            solution = undefined;
          } else if (isRateLimitExceededError(error)) {
            this.updateIsRateLimitExceeded(true);
          } else {
            this.updateIsRateLimitExceeded(false);
          }
        }
      }
      // Postprocess solution
      if (solution) {
        // Update Cache
        this.cache.set(hash, solution);

        const forwardingContexts = generateForwardingContexts(context, solution.items);
        forwardingContexts.forEach((entry) => {
          const forwardingContextHash = calculateCompletionContextHash(entry.context, this.documents);
          const forwardingSolution = new CompletionSolution();
          forwardingSolution.extraContext = solution?.extraContext ?? {};
          forwardingSolution.isCompleted = solution?.isCompleted ?? false;
          forwardingSolution.items = entry.items;
          this.cache.set(forwardingContextHash, forwardingSolution);
        });

        // postprocess: postCache
        solution.items = await postCacheProcess(solution.items, context, solution.extraContext, config.postprocess);
        if (signal.aborted) {
          throw signal.reason;
        }
      }
    } catch (error) {
      if (isCanceledError(error)) {
        this.logger.debug(`Providing completions canceled.`);
      } else {
        this.logger.error(`Providing completions failed.`, error);
      }
    }

    if (this.mutexAbortController === abortController) {
      this.mutexAbortController = undefined;
      this.updateIsFetching(false);
    }

    if (latencyStatsList.length > 0) {
      latencyStatsList.forEach((latencyStats) => {
        this.statisticTracker.addStatisticsEntry(latencyStats);

        if (latencyStats.latency !== undefined) {
          this.latencyTracker.add(latencyStats.latency);
        } else if (latencyStats.timeout) {
          this.latencyTracker.add(NaN);
        }
      });
      const statsResult = this.latencyTracker.calculateLatencyStatistics();
      const issue = analyzeMetrics(statsResult);
      switch (issue) {
        case "healthy":
          this.updateLatencyIssue(undefined);
          break;
        case "highTimeoutRate":
          this.updateLatencyIssue("highTimeoutRate");
          break;
        case "slowResponseTime":
          this.updateLatencyIssue("slowResponseTime");
          break;
      }
    }

    if (solution) {
      this.statisticTracker.addTriggerEntry({ triggerMode: manuallyTriggered ? "manual" : "auto" });
      this.logger.info(`Completed generating completions.`);
      this.logger.trace("Completion solution:", { items: solution.items });
      return { context, solution };
    }
    return null;
  }

  private async sendCompletionStatistics() {
    const report = this.statisticTracker.report();
    if (report["completion_request"]["count"] > 0) {
      await this.anonymousUsageLogger.event("AgentStats", { stats: report });
      this.statisticTracker.reset();
    }
  }
}
