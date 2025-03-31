import { QuickPickItem, window, QuickPickItemKind, CancellationTokenSource, ThemeIcon } from "vscode";
import { Deferred } from "../deferred";
import { Client } from "../lsp/client";
import { getLogger } from "../logger";

interface BranchQuickPickItem extends QuickPickItem {
  name: string;
}

const logger = getLogger("BranchQuickPick");

export class BranchQuickPick {
  quickPick = window.createQuickPick<BranchQuickPickItem>();
  private suggestedItems: BranchQuickPickItem[] = [];
  private resultDeferred = new Deferred<string | undefined>();
  private cancellationTokenSource = new CancellationTokenSource();
  private debounceTimer: NodeJS.Timeout | null = null;
  private cache = new Map<string, BranchQuickPickItem[]>();

  constructor(
    private readonly client: Client,
    private readonly repository: string,
  ) {
    this.quickPick.title = "Enter name to create new branch";
    this.quickPick.placeholder = "Type to filter branch names or create new";
    // Quick pick items are always sorted by label. issue: https://github.com/microsoft/vscode/issues/73904
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (this.quickPick as any).sortByLabel = false;

    this.quickPick.onDidChangeValue((value) => {
      this.quickPick.items = this.buildBranchList();

      if (this.debounceTimer) {
        clearTimeout(this.debounceTimer);
      }
      // Generate branch names even with a single character to improve responsiveness
      this.debounceTimer = setTimeout(() => {
        this.generateBranchNames(value);
      }, 300);
    });

    this.quickPick.onDidAccept(() => {
      this.handleAccept();
    });

    this.quickPick.onDidHide(() => {
      this.handleHidden();
    });
  }

  start() {
    this.generateBranchNames("");
    this.quickPick.show();
    return this.resultDeferred.promise;
  }

  private async generateBranchNames(input: string) {
    if (this.cancellationTokenSource) {
      this.cancellationTokenSource.cancel();
      this.cancellationTokenSource = new CancellationTokenSource();
    }

    const cacheKey = input.trim().toLowerCase();
    const cachedItems = this.cache.get(cacheKey);
    if (cachedItems) {
      this.suggestedItems = cachedItems;
      this.quickPick.items = this.buildBranchList();
      return;
    }

    if (this.quickPick) {
      this.quickPick.busy = true;
    }

    try {
      const result = await this.client.chat.generateBranchName(
        {
          repository: this.repository,
          input: input,
        },
        this.cancellationTokenSource.token,
      );

      if (result?.branchNames) {
        const uniqueBranches = result.branchNames.filter((name) => name.toLowerCase() !== input.toLowerCase());

        const branchItems = uniqueBranches.map((name) => ({
          label: name,
          name: name,
          iconPath: new ThemeIcon("sparkle"),
        }));

        this.cache.set(cacheKey, branchItems);
        logger.trace("Cached branch names for key:", cacheKey);

        this.suggestedItems = branchItems;
        this.quickPick.items = this.buildBranchList();
      }
    } catch (error) {
      if (!(error instanceof Error && error.name === "CancellationError")) {
        logger.error("Error generating branch names:", error);
      }
    } finally {
      if (this.quickPick) {
        this.quickPick.busy = false;
      }
    }
  }

  private buildBranchList(): BranchQuickPickItem[] {
    const input = this.quickPick.value;
    const list: BranchQuickPickItem[] = [];

    if (input) {
      list.push({
        label: input,
        name: input,
        iconPath: new ThemeIcon("add"),
        alwaysShow: true,
      });
    }

    list.push({
      label: "suggested names",
      name: "",
      kind: QuickPickItemKind.Separator,
    });
    list.push(...this.suggestedItems);

    return list;
  }

  private handleAccept() {
    const selection = this.quickPick.selectedItems[0];
    if (selection) {
      this.resultDeferred.resolve(selection.name);
      this.quickPick.hide();
    }
  }

  private handleHidden() {
    if (this.cancellationTokenSource) {
      this.cancellationTokenSource.cancel();
    }
    this.resultDeferred.resolve(undefined);
  }

  dispose() {
    if (this.cancellationTokenSource) {
      this.cancellationTokenSource.dispose();
    }
    if (this.quickPick) {
      this.quickPick.dispose();
    }
  }
}
