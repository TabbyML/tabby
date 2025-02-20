import type { Feature } from "../feature";
import type { ConfigData, PartialConfigData } from "./type";
import type { ConfigFile } from "./configFile";
import type {
  ClientCapabilities,
  ServerCapabilities,
  ClientProvidedConfig,
  Config as TabbyLspConfig,
} from "../protocol";
import type { TabbyServerProvidedConfig } from "../http/tabbyApiClient";
import { EventEmitter } from "events";
import { Connection } from "vscode-languageserver";
import deepEqual from "deep-equal";
import { deepmergeCustom } from "deepmerge-ts";
import { ConfigRequest, ConfigDidChangeNotification } from "../protocol";
import { defaultConfigData } from "./default";
import { getConfigFile } from "./configFile";
import { DataStore, StoredData } from "../dataStore";
import { isBlank } from "../utils/string";
import { getLogger } from "../logger";

function shouldBeNonBlankStringValue(meta: unknown) {
  if (typeof meta == "object" && meta !== null && "key" in meta && typeof meta.key === "string") {
    return ["endpoint", "token", "url", "authorization"].includes(meta.key);
  }
  return false;
}

const mergeFunction = deepmergeCustom({
  mergeOthers: (values, utils, meta) => {
    if (meta && shouldBeNonBlankStringValue(meta)) {
      const nonBlankStringValues = values.filter((value) => typeof value === "string" && !isBlank(value));
      if (nonBlankStringValues.length > 0) {
        return utils.defaultMergeFunctions.mergeOthers(nonBlankStringValues);
      } else {
        return "";
      }
    }
    return utils.actions.defaultMerge;
  },
});

function mergeConfig(
  base: ConfigData,
  configFile?: ConfigFile,
  clientProvided?: ClientProvidedConfig,
  serverProvided?: TabbyServerProvidedConfig,
): ConfigData {
  const configFileConfig: PartialConfigData = configFile?.config ?? {};
  const clientProvidedConfig: PartialConfigData = {};
  if (clientProvided?.server !== undefined) {
    clientProvidedConfig.server = clientProvided.server;
  }
  if (clientProvided?.proxy !== undefined) {
    clientProvidedConfig.proxy = clientProvided.proxy;
  }
  if (clientProvided?.anonymousUsageTracking !== undefined) {
    clientProvidedConfig.anonymousUsageTracking = clientProvided.anonymousUsageTracking;
  }
  const serverProvidedConfig: PartialConfigData = {};
  if (serverProvided?.disable_client_side_telemetry == true) {
    serverProvidedConfig.anonymousUsageTracking = { disable: true };
  }
  const merged = mergeFunction(base, configFileConfig, clientProvidedConfig, serverProvidedConfig) as ConfigData;

  // remove trailing slash from endpoint
  if (merged.server.endpoint) {
    merged.server.endpoint = merged.server.endpoint.replace(/\/+$/, "");
  }

  return merged;
}

export class Configurations extends EventEmitter implements Feature {
  private readonly logger = getLogger("Configurations");
  private readonly defaultConfig = defaultConfigData;

  private configFile: ConfigFile | undefined = undefined; // config from `~/.tabby-client/agent/config.toml`
  private clientProvided: ClientProvidedConfig = {}; // config from lsp client
  private serverProvided: TabbyServerProvidedConfig = {}; // config fetched from server and saved in dataStore
  private mergedConfig: ConfigData = defaultConfigData; // merged config from (default, configFile, clientProvided, serverProvided)

  private configForLsp: TabbyLspConfig = { server: defaultConfigData["server"] }; // config for lsp client

  private lspConnection: Connection | undefined = undefined;
  private clientCapabilities: ClientCapabilities | undefined = undefined;

  constructor(private readonly dataStore: DataStore) {
    super();
  }

  private pickStoredServerProvidedConfig(data: Partial<StoredData>): TabbyServerProvidedConfig {
    const mergedLocalConfig = mergeConfig(this.defaultConfig, this.configFile, this.clientProvided);
    const serverEndpoint = mergedLocalConfig.server.endpoint;
    return data.serverConfig?.[serverEndpoint] ?? {};
  }

  private update() {
    const old = this.mergedConfig;
    const merged = mergeConfig(this.defaultConfig, this.configFile, this.clientProvided, this.serverProvided);
    if (!deepEqual(old, merged)) {
      this.mergedConfig = merged;
      this.logger.trace("Updated configurations.", { config: merged });
      this.emit("updated", merged, old);

      const oldConfigForLsp = this.configForLsp;
      const configForLsp = { server: merged["server"] };
      if (!deepEqual(oldConfigForLsp, configForLsp)) {
        this.configForLsp = configForLsp;
        this.emit("configForLspUpdated", configForLsp, oldConfigForLsp);
      }
    }
  }

  async preInitialize(): Promise<void> {
    this.configFile = getConfigFile();
    if (this.configFile) {
      const configFile = this.configFile;
      await configFile.load();
      configFile.on("updated", async () => {
        this.serverProvided = this.pickStoredServerProvidedConfig(this.dataStore.data);
        this.update();
      });
      configFile.watch();
    }

    this.serverProvided = this.pickStoredServerProvidedConfig(this.dataStore.data);
    this.dataStore.on("updated", async (data: Partial<StoredData>) => {
      const serverProvidedConfig = this.pickStoredServerProvidedConfig(data);
      if (!deepEqual(serverProvidedConfig, this.serverProvided)) {
        this.serverProvided = serverProvidedConfig;
        this.update();
      }
    });

    this.update();
  }

  async initialize(
    connection: Connection,
    clientCapabilities: ClientCapabilities,
    clientProvidedConfig: ClientProvidedConfig,
  ): Promise<ServerCapabilities> {
    this.lspConnection = connection;
    this.clientCapabilities = clientCapabilities;

    this.updateClientProvidedConfig(clientProvidedConfig);

    connection.onDidChangeConfiguration(async (params) => {
      return this.updateClientProvidedConfig(params.settings);
    });
    connection.onRequest(ConfigRequest.type, async () => {
      return this.getConfigForLsp();
    });
    if (clientCapabilities.tabby?.configDidChangeListener) {
      this.on("configForLspUpdated", (config: TabbyLspConfig) => {
        connection.sendNotification(ConfigDidChangeNotification.type, config);
      });
    }

    return {};
  }

  async initialized(connection: Connection): Promise<void> {
    if (this.clientCapabilities?.tabby?.configDidChangeListener) {
      const config = this.getConfigForLsp();
      connection.sendNotification(ConfigDidChangeNotification.type, config);
    }
  }

  getClientProvidedConfig(): ClientProvidedConfig {
    return this.clientProvided;
  }

  getMergedConfig(): ConfigData {
    return this.mergedConfig;
  }

  getConfigForLsp(): TabbyLspConfig {
    return this.configForLsp;
  }

  async refreshClientProvidedConfig(): Promise<boolean> {
    if (this.lspConnection && this.clientCapabilities?.workspace?.configuration) {
      const config = await this.lspConnection.workspace.getConfiguration();
      this.updateClientProvidedConfig(config);
      return true;
    }
    return false;
  }

  private updateClientProvidedConfig(config: ClientProvidedConfig) {
    if (!deepEqual(config, this.clientProvided)) {
      const old = this.clientProvided;
      this.clientProvided = config;
      this.emit("clientProvidedConfigUpdated", config, old);
      this.serverProvided = this.pickStoredServerProvidedConfig(this.dataStore.data);
      this.update();
    }
  }

  async updateServerProvidedConfig(config: TabbyServerProvidedConfig, save: boolean = false) {
    if (!deepEqual(config, this.serverProvided)) {
      this.serverProvided = config;
      this.update();
    }
    if (save) {
      const mergedLocalConfig = mergeConfig(this.defaultConfig, this.configFile, this.clientProvided);
      const serverEndpoint = mergedLocalConfig.server.endpoint;
      if (!this.dataStore.data.serverConfig) {
        this.dataStore.data.serverConfig = {};
      }
      this.dataStore.data.serverConfig[serverEndpoint] = config;
      try {
        await this.dataStore.save();
      } catch (error) {
        this.logger.error("Failed to save server provided config.", error);
      }
    }
  }
}
