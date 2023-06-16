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
    const anonymousUsageLogger = new AnonymousUsageLogger();
    anonymousUsageLogger.dataStore = options.dataStore || dataStore;
    if (anonymousUsageLogger.dataStore) {
      try {
        await dataStore.load();
      } catch (_) {}
      if (typeof dataStore.data["anonymousId"] === "string") {
        anonymousUsageLogger.anonymousId = dataStore.data["anonymousId"];
      } else {
        anonymousUsageLogger.anonymousId = uuid();
        dataStore.data["anonymousId"] = anonymousUsageLogger.anonymousId;
        try {
          await dataStore.save();
        } catch (_) {}
      }
    } else {
      anonymousUsageLogger.anonymousId = uuid();
    }
    return anonymousUsageLogger;
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
