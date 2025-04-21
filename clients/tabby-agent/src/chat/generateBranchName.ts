import type { Connection, CancellationToken } from "vscode-languageserver";
import type { Feature } from "../feature";
import type { Configurations } from "../config";
import type { GitContextProvider } from "../contextProviders/git";
import {
  ServerCapabilities,
  ChatFeatureNotAvailableError,
  GenerateBranchNameRequest,
  GenerateBranchNameParams,
  GenerateBranchNameResult,
  GitDiffResult,
} from "../protocol";
import { isBlank, parseChatResponse, stringToRegExp } from "../utils/string";
import { MutexAbortError } from "../utils/error";
import { ChatFeature } from ".";

export class BranchNameGenerator implements Feature {
  private mutexAbortController: AbortController | undefined = undefined;

  constructor(
    private readonly chat: ChatFeature,
    private readonly configurations: Configurations,
    private readonly gitContextProvider: GitContextProvider,
  ) {}

  initialize(connection: Connection): ServerCapabilities {
    connection.onRequest(GenerateBranchNameRequest.type, async (params, token) => {
      return this.generateBranchName(params, token);
    });
    return {};
  }

  async generateBranchName(
    params: GenerateBranchNameParams,
    token: CancellationToken,
  ): Promise<GenerateBranchNameResult | null> {
    if (!this.chat.isAvailable()) {
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

    const { repository, input } = params;
    let diffResult: GitDiffResult | undefined | null = undefined;
    diffResult = await this.gitContextProvider.diff({ repository, cached: true }, token);
    if (
      !diffResult?.diff ||
      (typeof diffResult.diff === "string" && isBlank(diffResult.diff)) ||
      (Array.isArray(diffResult.diff) && isBlank(diffResult.diff.join("")))
    ) {
      diffResult = await this.gitContextProvider.diff({ repository, cached: false }, token);
    }

    if (!diffResult || !diffResult.diff) {
      return null;
    }

    const config = this.configurations.getMergedConfig();
    const { maxDiffLength, promptTemplate } = config.chat.generateBranchName;

    let userPrompt = promptTemplate;
    if (input && input.trim() && userPrompt.includes("{{input}}")) {
      userPrompt = userPrompt.replace(/{{input}}/g, input.trim());
    }

    const responseMatcher = "^[a-z0-9][a-z0-9-]*[a-z0-9]$";

    const diff = diffResult.diff;
    let splitDiffs: string[];
    if (typeof diff === "string") {
      splitDiffs = diff.split(/\n(?=diff)/);
    } else {
      splitDiffs = diff;
    }
    let selectedDiff = "";
    for (const item of splitDiffs) {
      if (selectedDiff.length + item.length > maxDiffLength) {
        break;
      }
      selectedDiff += item + "\n";
    }
    if (isBlank(selectedDiff)) {
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
        content: userPrompt.replace("{{diff}}", selectedDiff),
      },
    ];
    const readableStream = await this.chat.tabbyApiClient.fetchChatStream(
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

    let branchNamesContent = responseMessage;
    const branchNamesMatch = /<BRANCHNAMES>([\s\S]*?)<\/BRANCHNAMES>/i.exec(responseMessage);
    if (branchNamesMatch && branchNamesMatch[1]) {
      branchNamesContent = branchNamesMatch[1].trim();
    }

    const matcherReg = stringToRegExp(responseMatcher);

    let branchNames = branchNamesContent
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
      .map((line) => {
        const match = matcherReg.exec(line);
        return match
          ? match[0]
          : line
              .toLowerCase()
              .replace(/[^a-z0-9]+/g, "-")
              .replace(/^-+|-+$/g, "");
      })
      .filter((branchName) => branchName.length > 0)
      .slice(0, 5);

    if (input && input.trim()) {
      const inputPrefix = input.trim().toLowerCase();

      const matchingBranches = branchNames.filter((name) => name.startsWith(inputPrefix));

      if (matchingBranches.length === 0) {
        branchNames = branchNames.map((name) => {
          if (inputPrefix.endsWith("-") && name.startsWith("-")) {
            return inputPrefix + name.substring(1);
          }
          return inputPrefix + (inputPrefix.endsWith("-") || name.startsWith("-") ? "" : "-") + name;
        });
      } else {
        branchNames = matchingBranches;
      }
    }

    while (branchNames.length < 3 && branchNames.length > 0) {
      branchNames.push(branchNames[0] + "-alt");
    }

    return { branchNames };
  }
}
