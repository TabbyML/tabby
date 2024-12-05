import type { Connection, CancellationToken } from "vscode-languageserver";
import type { Feature } from "../feature";
import type { Configurations } from "../config";
import type { TabbyApiClient } from "../http/tabbyApiClient";
import type { GitContextProvider } from "../git";
import {
  ServerCapabilities,
  ChatFeatureNotAvailableError,
  GenerateCommitMessageRequest,
  GenerateCommitMessageParams,
  GenerateCommitMessageResult,
  GitDiffResult,
} from "../protocol";
import { isBlank, parseChatResponse, stringToRegExp } from "../utils/string";
import { MutexAbortError } from "../utils/error";

export class CommitMessageGenerator implements Feature {
  private mutexAbortController: AbortController | undefined = undefined;

  constructor(
    private readonly configurations: Configurations,
    private readonly tabbyApiClient: TabbyApiClient,
    private readonly gitContextProvider: GitContextProvider,
  ) {}

  initialize(connection: Connection): ServerCapabilities {
    connection.onRequest(GenerateCommitMessageRequest.type, async (params, token) => {
      return this.generateCommitMessage(params, token);
    });
    return {};
  }

  async generateCommitMessage(
    params: GenerateCommitMessageParams,
    token: CancellationToken,
  ): Promise<GenerateCommitMessageResult | null> {
    if (!this.tabbyApiClient.isChatApiAvailable()) {
      throw {
        name: "ChatFeatureNotAvailableError",
        message: "Chat feature not available",
      } as ChatFeatureNotAvailableError;
    }

    if (token.isCancellationRequested) {
      return null;
    }
    if (this.mutexAbortController && !this.mutexAbortController.signal.aborted) {
      this.mutexAbortController.abort(new MutexAbortError());
    }
    this.mutexAbortController = new AbortController();
    token.onCancellationRequested(() => this.mutexAbortController?.abort());

    const { repository } = params;
    let diffResult: GitDiffResult | undefined | null = undefined;
    diffResult = await this.gitContextProvider.diff({ repository, cached: true }, token);
    if (
      !diffResult?.diff ||
      (typeof diffResult.diff === "string" && isBlank(diffResult.diff)) ||
      (Array.isArray(diffResult.diff) && isBlank(diffResult.diff.join("")))
    ) {
      // Use uncached diff if cached diff is empty
      diffResult = await this.gitContextProvider.diff({ repository, cached: false }, token);
    }

    if (!diffResult || !diffResult.diff) {
      return null;
    }

    const config = this.configurations.getMergedConfig();
    const { maxDiffLength, promptTemplate, responseMatcher } = config.chat.generateCommitMessage;

    // select diffs from the list to generate a prompt under the prompt size limit
    const diff = diffResult.diff;
    let splitDiffs: string[];
    if (typeof diff === "string") {
      splitDiffs = diff.split(/\n(?=diff)/);
    } else {
      splitDiffs = diff;
    }
    let selectedDiff = "";
    for (const item of splitDiffs) {
      if (selectedDiff.length + item.length < maxDiffLength) {
        selectedDiff += item + "\n";
      }
    }
    if (isBlank(selectedDiff)) {
      // This may happen when all separated diffs are larger than the limit.
      if (typeof diff === "string") {
        selectedDiff = diff.substring(0, maxDiffLength);
      } else {
        selectedDiff = diff.join("\n").substring(0, maxDiffLength);
      }
    }
    if (isBlank(selectedDiff)) {
      return null;
    }

    const messages: { role: "user"; content: string }[] = [
      {
        role: "user",
        content: promptTemplate.replace("{{diff}}", selectedDiff),
      },
    ];
    const readableStream = await this.tabbyApiClient.fetchChatStream(
      {
        messages,
        model: "",
        stream: true,
      },
      this.mutexAbortController.signal,
    );
    if (!readableStream) {
      return null;
    }

    const responseMessage = await parseChatResponse(readableStream);
    const matcherReg = stringToRegExp(responseMatcher);
    const match = matcherReg.exec(responseMessage);

    const commitMessage = (match ? match[0] : responseMessage).trim();
    return { commitMessage };
  }
}
