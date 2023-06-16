import { name as agentName, version as agentVersion } from "../package.json";
import { CloudApi } from "./cloud";
import { v4 as uuid } from "uuid";
import { isBrowser } from "./utils";
import { rootLogger } from "./logger";
import { dataStore, DataStore } from "./dataStore";

export class AnonymousUsageLogger {
  private anonymousUsageTrackingApi = new CloudApi({ BASE: "https://app.tabbyml.com/api" });
  private logger = rootLogger.child({ component: "AnonymousUsage" });
  private systemData = {
    agent: `${agentName}, ${agentVersion}`,
    browser: isBrowser ? navigator?.userAgent || "browser" : undefined,
    node: isBrowser
      ? undefined
      : `${process.version} ${process.platform} ${require("os").arch()} ${require("os").release()}`,
  };
  private dataStore: DataStore | null = null;
  private anonymousId: string;

  disabled: boolean;

  private constructor() {}

  static async create(options: { dataStore: DataStore }): Promise<AnonymousUsageLogger> {
    const logger = new AnonymousUsageLogger();
    logger.dataStore = options.dataStore || dataStore;
    await logger.checkAnonymousId();
    return logger;
  }

  private async checkAnonymousId() {
    if (this.dataStore) {
      try {
        await this.dataStore.load();
      } catch (error) {
        this.logger.debug({ error }, "Error when loading anonymousId");
      }
      if (typeof this.dataStore.data["anonymousId"] === "string") {
        this.anonymousId = this.dataStore.data["anonymousId"];
      } else {
        this.anonymousId = uuid();
        this.dataStore.data["anonymousId"] = this.anonymousId;
        try {
          await this.dataStore.save();
        } catch (error) {
          this.logger.debug({ error }, "Error when saving anonymousId");
        }
      }
    } else {
      this.anonymousId = uuid();
    }
  }

  async event(event: string, data: any) {
    if (this.disabled) {
      return;
    }
    await this.anonymousUsageTrackingApi.api
      .usage({
        distinctId: this.anonymousId,
        event,
        properties: {
          ...this.systemData,
          ...data,
        },
      })
      .catch((error) => {
        this.logger.error({ error }, "Error when sending anonymous usage data");
      });
  }
}
