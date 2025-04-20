import { Connection, Location } from "vscode-languageserver";
import { Feature } from "../feature";
import { DidChangeActiveEditorNotification, DidChangeActiveEditorParams, ServerCapabilities } from "../protocol";
import { Configurations } from "../config";
import { LRUCache } from "lru-cache";
import { intersectionRange } from "../utils/range";
import { ConfigData } from "../config/type";
import deepEqual from "deep-equal";

export class EditorVisibleRangesTracker implements Feature {
  private history: LRUCache<number, Location> | undefined = undefined;
  private version = 0;

  constructor(private readonly configurations: Configurations) {}

  initialize(connection: Connection): ServerCapabilities | Promise<ServerCapabilities> {
    this.setup();

    this.configurations.on("updated", (config: ConfigData, oldConfig: ConfigData) => {
      if (!deepEqual(pickConfig(config), pickConfig(oldConfig))) {
        this.shutdown();
        this.setup();
      }
    });

    connection.onNotification(DidChangeActiveEditorNotification.type, (param: DidChangeActiveEditorParams) => {
      this.updateHistory(param);
    });
    return {};
  }

  shutdown() {
    this.history = undefined;
  }

  private setup() {
    const config = pickConfig(this.configurations.getMergedConfig());
    if (config.enabled) {
      this.history = new LRUCache<number, Location>({
        max: 1000,
        ttl: 5 * 60 * 1000, // 5 minutes
      });
    }
  }

  private updateHistory(param: DidChangeActiveEditorParams) {
    if (this.history) {
      this.version++;
      this.history.set(this.version, param.activeEditor);
    }
  }

  getVersion(): number {
    return this.version;
  }

  async getHistoryRanges(options?: { max?: number; excludedUris?: string[] }): Promise<Location[] | undefined> {
    if (!this.history) {
      return undefined;
    }

    const result: Location[] = [];
    for (const item of this.history.values()) {
      if (options?.max && result.length >= options?.max) {
        break;
      }
      const location = await item;
      if (location) {
        if (options?.excludedUris?.includes(location.uri)) {
          continue;
        }

        const foundIntersection = result.find(
          (r) => r.uri === location.uri && intersectionRange(r.range, location.range),
        );
        if (!foundIntersection) {
          result.push(location);
        }
      }
    }
    return result;
  }
}

function pickConfig(configData: ConfigData) {
  return configData.completion.prompt.collectSnippetsFromRecentOpenedFiles;
}
