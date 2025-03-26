import { QuickPickItem, window, QuickPickItemKind, CancellationTokenSource, ThemeIcon } from "vscode";
import { Deferred } from "./util";
import { Client } from "../lsp/client";
import { CancellationToken } from "vscode-languageclient";
import { getLogger } from "../logger";

interface BranchQuickPickItem extends QuickPickItem {
  name: string;
}

// 简化缓存条目，移除过期时间
interface CacheEntry {
  items: BranchQuickPickItem[];
}

export class BranchQuickPick {
  quickPick = window.createQuickPick<BranchQuickPickItem>();
  private resultDeferred = new Deferred<string | undefined>();
  private cancellationTokenSource = new CancellationTokenSource();
  private items: BranchQuickPickItem[] = [];
  private debounceTimer: NodeJS.Timeout | null = null;
  private isGenerating = false;
  private cache: Map<string, CacheEntry>;

  constructor(
    private readonly client: Client,
    private readonly repository: string,
    private readonly token?: CancellationToken,
  ) {
    this.cache = new Map<string, CacheEntry>();

    if (this.token) {
      this.token.onCancellationRequested(() => {
        if (this.cancellationTokenSource) {
          this.cancellationTokenSource.cancel();
        }
        this.dispose();
      });
    }

    this.setupQuickPick();
  }

  private getCachedItems(key: string): BranchQuickPickItem[] | null {
    const entry = this.cache.get(key);
    if (!entry) {
      return null;
    }
    return entry.items;
  }

  private setCachedItems(key: string, items: BranchQuickPickItem[]) {
    this.cache.set(key, { items });
  }

  private async generateBranchNames(value: string) {
    // Check if parent token is already cancelled
    if (this.token?.isCancellationRequested) {
      return;
    }

    if (this.isGenerating && this.cancellationTokenSource) {
      this.cancellationTokenSource.cancel();
      this.cancellationTokenSource = new CancellationTokenSource();
    }

    const cacheKey = value.trim().toLowerCase();
    const cachedItems = this.getCachedItems(cacheKey);
    if (cachedItems) {
      this.items = cachedItems;
      this.updateBranchList(this.quickPick.value);
      return;
    }

    this.isGenerating = true;
    if (this.quickPick) {
      this.quickPick.busy = true;
    }

    try {
      const result = (await this.client.chat.generateBranchName(
        {
          repository: this.repository,
          input: value,
        },
        this.cancellationTokenSource.token,
      )) as unknown as { branchNames: string[] } | null;

      if (result?.branchNames) {
        const userInput = value.toLowerCase();
        const uniqueBranches = result.branchNames.filter((name) => name.toLowerCase() !== userInput);

        const branchItems = uniqueBranches.map((name) => ({
          label: name,
          description: "Generated branch name",
          name: name,
          iconPath: new ThemeIcon("git-branch"),
        }));

        this.setCachedItems(cacheKey, branchItems);
        getLogger().info("Cached branch names for key:", cacheKey);

        this.items = branchItems;
        this.updateBranchList(this.quickPick.value);
      }
    } catch (error) {
      if (!(error instanceof Error && error.name === "CancellationError")) {
        console.error("Error generating branch names:", error);
      }
    } finally {
      this.isGenerating = false;
      if (this.quickPick) {
        this.quickPick.busy = false;
      }
    }
  }

  private setupQuickPick() {
    this.quickPick.title = "Enter branch name";
    this.quickPick.placeholder = "Type to filter branches or create new";
    this.quickPick.matchOnDescription = true;
    this.quickPick.items = this.getBranchList("");

    this.quickPick.onDidChangeValue((value) => {
      this.updateBranchList(value);
      getLogger().info("current value:", value);

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
    // Check if already cancelled
    if (this.token?.isCancellationRequested) {
      return Promise.resolve(undefined);
    }

    const emptyKey = "";
    const cachedItems = this.getCachedItems(emptyKey);

    if (cachedItems) {
      getLogger().info("Using cached empty branch names");
      this.items = cachedItems;
      this.updateBranchList("");
    } else {
      this.generateBranchNames("");
    }

    this.quickPick.show();
    return this.resultDeferred.promise;
  }

  private updateBranchList(query: string) {
    // Remember current selection before updating items
    const currentSelection = this.quickPick.selectedItems.length > 0 ? this.quickPick.selectedItems[0]?.name : null;

    // Update the items
    this.quickPick.items = this.getBranchList(query);

    // If there was a selection, try to restore it
    if (currentSelection !== null) {
      const selectedIndex = this.quickPick.items.findIndex((item) => item.name === currentSelection);

      if (selectedIndex >= 0) {
        // Need to wait for VS Code to update the UI
        setTimeout(() => {
          const item = this.quickPick.items[selectedIndex];
          if (this.quickPick && item) {
            this.quickPick.activeItems = [item];
          }
        }, 0);
      }
    }
  }

  private getBranchList(query: string): BranchQuickPickItem[] {
    const list: BranchQuickPickItem[] = [];

    if (query) {
      list.push({
        label: `Create branch "${query}"`,
        name: query,
        iconPath: new ThemeIcon("add"),
        alwaysShow: true,
      });

      list.push({
        label: "",
        name: "",
        kind: QuickPickItemKind.Separator,
      });
    }

    const lowerQuery = query.toLowerCase();
    const filteredBranches = this.items.filter((item) => {
      if (query && item.name.toLowerCase() === lowerQuery) {
        return false;
      }
      return item.name.toLowerCase().includes(lowerQuery);
    });

    list.push(...filteredBranches);

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
