import type { paths as CloudApi } from "./http/cloudApi";
import type { ClientInfo, ClientProvidedConfig } from "./protocol";
import type { DataStore } from "./dataStore";
import type { Configurations } from "./config";
import type { ConfigData } from "./config/type";
import os from "os";
import createClient from "openapi-fetch";
import { setProperty } from "dot-prop";
import deepEqual from "deep-equal";
import { v4 as uuid } from "uuid";
import { name as agentName, version as agentVersion } from "../package.json";
import { isBrowser } from "./env";
import { ProxyConfig, createProxyForUrl } from "./http/proxy";
import { getLogger } from "./logger";
import { isBlank } from "./utils/string";

export class AnonymousUsageLogger {
  private readonly logger = getLogger("Telemetry");
  private readonly systemData = {
    agent: `${agentName}, ${agentVersion}`,
    browser: isBrowser ? navigator?.userAgent || "browser" : undefined,
    node: isBrowser ? undefined : `${process.version} ${process.platform} ${os.arch()} ${os.release()}`,
  };

  private api: ReturnType<typeof createClient<CloudApi>> | undefined;
  private clientInfoProperties: Record<string, any> = {};
  private userProperties: Record<string, any> = {};
  private shouldUpdateUserProperties = false;

  private anonymousId: string | undefined = undefined;
  private loggedUniqueEvents: string[] = [];

  private disabled: boolean = false;

  constructor(
    private readonly dataStore: DataStore,
    private readonly configurations: Configurations,
  ) {}

  async initialize(clientInfo: ClientInfo | undefined) {
    const config = this.configurations.getMergedConfig();
    const endpoint = "https://app.tabbyml.com/api";
    const proxyConfigs: ProxyConfig[] = [{ fromEnv: true }];
    if (!isBlank(config.proxy.url)) {
      proxyConfigs.unshift(config.proxy);
    }
    this.api = createClient<CloudApi>({
      baseUrl: endpoint,
      /** dispatcher do not exist in {@link RequestInit} in browser env. */
      /* @ts-expect-error TS-2353 */
      dispatcher: createProxyForUrl(endpoint, proxyConfigs),
    });

    if (typeof this.dataStore.data["anonymousId"] === "string") {
      this.anonymousId = this.dataStore.data["anonymousId"];
    } else {
      this.anonymousId = uuid();
      this.dataStore.data["anonymousId"] = this.anonymousId;
      try {
        await this.dataStore.save();
      } catch (error) {
        this.logger.error("Failed to save anonymous Id.", error);
      }
    }

    this.disabled = this.configurations.getMergedConfig().anonymousUsageTracking.disable;
    this.logger.info("Anonymous usage tracking disabled: " + this.disabled);
    this.configurations.on("updated", (config: ConfigData) => {
      if (this.disabled !== config.anonymousUsageTracking.disable) {
        this.disabled = config.anonymousUsageTracking.disable;
        this.logger.info("Update anonymous usage tracking disabled: " + this.disabled);
      }
    });

    const clientProvidedConfig = this.configurations.getClientProvidedConfig();
    this.updateUserProperties(clientInfo, clientProvidedConfig);
    this.configurations.on("clientProvidedConfigUpdated", (clientProvidedConfig: ClientProvidedConfig) => {
      this.updateUserProperties(clientInfo, clientProvidedConfig);
    });

    this.updateClientInfoProperties(clientInfo);
  }

  async uniqueEvent(event: string, data: Record<string, any> = {}) {
    await this.event(event, data, true);
  }

  async event(event: string, data: Record<string, any> = {}, unique = false) {
    if (this.disabled || !this.anonymousId || !this.api) {
      return;
    }
    if (unique && this.loggedUniqueEvents.includes(event)) {
      return;
    }
    if (unique) {
      this.loggedUniqueEvents.push(event);
    }
    const properties = {
      ...this.systemData,
      ...this.clientInfoProperties,
      ...data,
    };
    if (this.shouldUpdateUserProperties) {
      setProperty(properties, "$set", this.userProperties);
      this.shouldUpdateUserProperties = false;
    }
    try {
      const request = {
        distinctId: this.anonymousId,
        event,
        properties,
      };
      this.logger.trace("Sending anonymous usage data.", { request });
      await this.api.POST("/usage", {
        body: request,
      });
    } catch (error) {
      this.logger.error("Failed to send anonymous usage data.", error);
      this.loggedUniqueEvents = this.loggedUniqueEvents.filter((e) => e !== event);
    }
  }

  private updateUserProperties(clientInfo: ClientInfo | undefined, clientProvidedConfig: ClientProvidedConfig) {
    const clientType = this.getClientType(clientInfo);
    const properties = {
      [clientType]: {
        triggerMode: clientProvidedConfig?.inlineCompletion?.triggerMode,
        keybindings: clientProvidedConfig?.keybindings,
      },
    };
    if (!deepEqual(properties, this.userProperties)) {
      this.userProperties = properties;
      this.shouldUpdateUserProperties = true;
    }
  }

  private updateClientInfoProperties(clientInfo: ClientInfo | undefined) {
    const properties = {
      client: `${clientInfo?.name} ${clientInfo?.version ?? ""}`,
      ide: {
        name: clientInfo?.name,
        version: clientInfo?.version,
      },
      tabby_plugin: clientInfo?.tabbyPlugin ?? {
        name: agentName,
        version: agentVersion,
      },
    };
    if (!deepEqual(properties, this.clientInfoProperties)) {
      this.clientInfoProperties = properties;
    }
  }

  private getClientType(clientInfo: ClientInfo | undefined): string {
    if (!clientInfo) {
      return "unknown";
    }
    if (clientInfo.tabbyPlugin?.name.includes("vscode")) {
      return "vscode";
    } else if (clientInfo.tabbyPlugin?.name.includes("intellij")) {
      return "intellij";
    } else if (clientInfo.tabbyPlugin?.name.includes("vim")) {
      return "vim";
    }
    return clientInfo.name;
  }
}
