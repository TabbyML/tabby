import os from "os";
import createClient from "openapi-fetch";
import { setProperty } from "dot-prop";
import { v4 as uuid } from "uuid";
import type { paths as CloudApi } from "./types/cloudApi";
import { name as agentName, version as agentVersion } from "../package.json";
import { isBrowser } from "./env";
import { rootLogger } from "./logger";
import { dataStore, DataStore } from "./dataStore";

export class AnonymousUsageLogger {
  private anonymousUsageTrackingApi = createClient<CloudApi>({ baseUrl: "https://app.tabbyml.com/api" });
  private logger = rootLogger.child({ component: "AnonymousUsage" });
  private systemData = {
    agent: `${agentName}, ${agentVersion}`,
    browser: isBrowser ? navigator?.userAgent || "browser" : undefined,
    node: isBrowser ? undefined : `${process.version} ${process.platform} ${os.arch()} ${os.release()}`,
  };
  private sessionProperties: Record<string, any> = {};
  private userProperties: Record<string, any> = {};
  private userPropertiesUpdated = false;
  private emittedUniqueEvent: string[] = [];
  private dataStore?: DataStore;
  private anonymousId?: string;
  disabled: boolean = false;

  async init(options?: { dataStore?: DataStore }) {
    this.dataStore = options?.dataStore || dataStore;
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

  /**
   * Set properties to be sent with every event in this session.
   */
  setSessionProperties(key: string, value: any) {
    setProperty(this.sessionProperties, key, value);
  }

  /**
   * Set properties which will be bind to the user.
   */
  setUserProperties(key: string, value: any) {
    setProperty(this.userProperties, key, value);
    this.userPropertiesUpdated = true;
  }

  async uniqueEvent(event: string, data: Record<string, any> = {}) {
    await this.event(event, data, true);
  }

  async event(event: string, data: Record<string, any> = {}, unique = false) {
    if (this.disabled || !this.anonymousId) {
      return;
    }
    if (unique && this.emittedUniqueEvent.includes(event)) {
      return;
    }
    if (unique) {
      this.emittedUniqueEvent.push(event);
    }
    const properties = {
      ...this.systemData,
      ...this.sessionProperties,
      ...data,
    };
    if (this.userPropertiesUpdated) {
      setProperty(properties, "$set", this.userProperties);
      this.userPropertiesUpdated = false;
    }
    try {
      await this.anonymousUsageTrackingApi.POST("/usage", {
        body: {
          distinctId: this.anonymousId,
          event,
          properties,
        },
      });
    } catch (error) {
      this.logger.error({ error }, "Error when sending anonymous usage data");
    }
  }
}
