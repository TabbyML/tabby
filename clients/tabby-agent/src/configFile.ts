import { EventEmitter } from "events";
import path from "path";
import os from "os";
import fs from "fs-extra";
import toml from "toml";
import chokidar from "chokidar";
import deepEqual from "deep-equal";
import { getProperty, deleteProperty } from "dot-prop";
import type { PartialAgentConfig } from "./AgentConfig";
import { isBrowser } from "./env";
import { rootLogger } from "./logger";

const configTomlTemplate = `## Tabby agent configuration file

## Online documentation: https://tabby.tabbyml.com/docs/extensions/configurations
## You can uncomment and edit the values below to change the default settings.
## Configurations in this file have lower priority than the IDE settings.

## Server
## You can set the server endpoint here and an optional authentication token if required.
# [server]
# endpoint = "http://localhost:8080" # http or https URL
# token = "your-token-here" # if token is set, request header Authorization = "Bearer $token" will be added automatically

## You can add custom request headers.
# [server.requestHeaders]
# Header1 = "Value1" # list your custom headers here
# Header2 = "Value2" # values can be strings, numbers or booleans

## Completion
## (Since 1.1.0) You can set the completion request timeout here.
## Note that there is also a timeout config at the server side.
# [completion]
# timeout = 4000 # 4s

## Logs
## You can set the log level here. The log file is located at ~/.tabby-client/agent/logs/.
# [logs]
# level = "silent" # "silent" or "error" or "debug"

## Anonymous usage tracking
## Tabby collects anonymous usage data and sends it to the Tabby team to help improve our products.
## Your code, generated completions, or any sensitive information is never tracked or sent.
## For more details on data collection, see https://tabby.tabbyml.com/docs/extensions/configurations#usage-collection
## Your contribution is greatly appreciated. However, if you prefer not to participate, you can disable anonymous usage tracking here.
# [anonymousUsageTracking]
# disable = false # set to true to disable

`;

const typeCheckSchema: Record<string, string> = {
  server: "object",
  "server.endpoint": "string",
  "server.token": "string",
  "server.requestHeaders": "object",
  "server.requestTimeout": "number",
  completion: "object",
  "completion.prompt": "object",
  "completion.prompt.experimentalStripAutoClosingCharacters": "boolean",
  "completion.prompt.maxPrefixLines": "number",
  "completion.prompt.maxSuffixLines": "number",
  "completion.prompt.clipboard": "object",
  "completion.prompt.clipboard.minChars": "number",
  "completion.prompt.clipboard.maxChars": "number",
  "completion.debounce": "object",
  "completion.debounce.mode": "string",
  "completion.debounce.interval": "number",
  "completion.timeout": "number",
  postprocess: "object",
  "postprocess.limitScopeByIndentation": "object",
  "postprocess.limitScopeByIndentation.experimentalKeepBlockScopeWhenCompletingLine": "boolean",
  logs: "object",
  "logs.level": "string",
  anonymousUsageTracking: "object",
  "anonymousUsageTracking.disable": "boolean",
};

function validateConfig(config: PartialAgentConfig): PartialAgentConfig {
  for (const [key, type] of Object.entries(typeCheckSchema)) {
    if (typeof getProperty(config, key) !== type) {
      deleteProperty(config, key);
    }
  }
  return config;
}

class ConfigFile extends EventEmitter {
  private data: PartialAgentConfig = {};
  private watcher?: chokidar.FSWatcher;
  private logger = rootLogger.child({ component: "ConfigFile" });

  constructor(private readonly filepath: string) {
    super();
  }

  get config(): PartialAgentConfig {
    return this.data;
  }

  async load() {
    try {
      const fileContent = await fs.readFile(this.filepath, "utf8");
      const data = toml.parse(fileContent);
      // If the config file contains no value, overwrite it with the new template.
      if (Object.keys(data).length === 0 && fileContent.trim() !== configTomlTemplate.trim()) {
        await this.createTemplate();
        return;
      }
      this.data = validateConfig(data);
    } catch (error) {
      if (error instanceof Error && "code" in error && error.code === "ENOENT") {
        await this.createTemplate();
      } else {
        this.logger.error({ error }, "Failed to load config file");
      }
    }
  }

  watch() {
    this.watcher = chokidar.watch(this.filepath, {
      interval: 1000,
    });
    const onChanged = async () => {
      const oldData = this.data;
      await this.load();
      if (!deepEqual(oldData, this.data)) {
        super.emit("updated", this.data);
      }
    };
    this.watcher.on("add", onChanged);
    this.watcher.on("change", onChanged);
  }

  private async createTemplate() {
    try {
      await fs.outputFile(this.filepath, configTomlTemplate);
    } catch (error) {
      this.logger.error({ error }, "Failed to create config template file");
    }
  }
}

const configFilePath = path.join(os.homedir(), ".tabby-client", "agent", "config.toml");
export const configFile = isBrowser ? undefined : new ConfigFile(configFilePath);
