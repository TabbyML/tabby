import type { PartialConfigData } from "./type";
import { EventEmitter } from "events";
import path from "path";
import os from "os";
import fs from "fs-extra";
import toml from "toml";
import chokidar from "chokidar";
import deepEqual from "deep-equal";
import { getProperty, deleteProperty } from "dot-prop";
import { isBrowser } from "../env";
import { getLogger } from "../logger";

const configTomlTemplate = `## Tabby agent configuration file

## Online documentation: https://tabby.tabbyml.com/docs/extensions/configurations
## You can uncomment and edit the values below to change the default settings.
## Configurations in this file have lower priority than the IDE settings.

## Server
## You can set the server endpoint and token here.
# [server]
# endpoint = "http://localhost:8080" # http or https URL
# token = "your-token-here" # if set, request header Authorization = "Bearer $token" will be added

## You can add custom request headers.
# [server.requestHeaders]
# Header1 = "Value1" # list your custom headers here
# Header2 = "Value2" # values can be strings, numbers or booleans

## Proxy
## You can specify an optional http/https proxy when required, overrides environment variable settings.
# [proxy]
# url = "http://your-proxy-server" # the URL of the proxy

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
  proxy: "object",
  "proxy.url": "string",
  completion: "object",
  "completion.prompt": "object",
  "completion.prompt.maxPrefixLines": "number",
  "completion.prompt.maxSuffixLines": "number",
  "completion.prompt.fillDeclarations": "object",
  "completion.prompt.fillDeclarations.enabled": "boolean",
  "completion.prompt.fillDeclarations.maxSnippets": "number",
  "completion.prompt.fillDeclarations.maxChars": "number",
  "completion.prompt.collectSnippetsFromRecentChangedFiles": "object",
  "completion.prompt.collectSnippetsFromRecentChangedFiles.enabled": "boolean",
  "completion.prompt.collectSnippetsFromRecentChangedFiles.maxSnippets": "number",
  "completion.prompt.collectSnippetsFromRecentChangedFiles.indexing": "object",
  "completion.prompt.collectSnippetsFromRecentChangedFiles.indexing.checkingChangesInterval": "number",
  "completion.prompt.collectSnippetsFromRecentChangedFiles.indexing.changesDebouncingInterval": "number",
  "completion.prompt.collectSnippetsFromRecentChangedFiles.indexing.prefixLines": "number",
  "completion.prompt.collectSnippetsFromRecentChangedFiles.indexing.suffixLines": "number",
  "completion.prompt.collectSnippetsFromRecentChangedFiles.indexing.maxChunks": "number",
  "completion.prompt.collectSnippetsFromRecentChangedFiles.indexing.chunkSize": "number",
  "completion.prompt.collectSnippetsFromRecentChangedFiles.indexing.overlapLines": "number",
  "completion.prompt.clipboard": "object",
  "completion.prompt.clipboard.minChars": "number",
  "completion.prompt.clipboard.maxChars": "number",
  "completion.debounce": "object",
  "completion.debounce.mode": "string",
  "completion.debounce.interval": "number",
  "completion.solution": "object",
  "completion.solution.maxItems": "number",
  "completion.solution.maxTries": "number",
  "completion.solution.temperature": "number",
  chat: "object",
  "chat.edit": "object",
  "chat.generateCommitMessage": "object",
  "chat.generateCommitMessage.maxDiffLength": "number",
  "chat.generateCommitMessage.promptTemplate": "string",
  logs: "object",
  "logs.level": "string",
  tls: "object",
  "tls.caCerts": "string",
  anonymousUsageTracking: "object",
  "anonymousUsageTracking.disable": "boolean",
};

function validateConfig(config: PartialConfigData): PartialConfigData {
  for (const [key, type] of Object.entries(typeCheckSchema)) {
    if (typeof getProperty(config, key) !== type) {
      deleteProperty(config, key);
    }
  }
  return config;
}

export class ConfigFile extends EventEmitter {
  private data: PartialConfigData = {};
  private watcher?: chokidar.FSWatcher;
  private logger = getLogger("ConfigFile");

  constructor(private readonly filepath: string) {
    super();
  }

  get config(): PartialConfigData {
    return this.data;
  }

  async load() {
    try {
      const fileContent = await fs.readFile(this.filepath, "utf8");
      const data = toml.parse(fileContent);
      this.data = validateConfig(data);
    } catch (error) {
      if (error instanceof Error && "code" in error && error.code === "ENOENT") {
        this.logger.info("Config file not exist, creating template config file.");
        await this.createTemplate();
      } else {
        this.logger.error("Failed to load config file.", error);
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
        this.emit("updated", this.data, oldData);
      }
    };
    this.watcher.on("add", onChanged);
    this.watcher.on("change", onChanged);
  }

  private async createTemplate() {
    try {
      await fs.outputFile(this.filepath, configTomlTemplate);
    } catch (error) {
      this.logger.error("Failed to create config template file.", error);
    }
  }
}

export function getConfigFile(): ConfigFile | undefined {
  const configFilePath = path.join(os.homedir(), ".tabby-client", "agent", "config.toml");
  return isBrowser ? undefined : new ConfigFile(configFilePath);
}
