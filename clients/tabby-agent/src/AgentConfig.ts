import { isBrowser } from "./env";
import { getProperty, deleteProperty } from "dot-prop";

export type AgentConfig = {
  server: {
    endpoint: string;
    token: string;
    requestHeaders: Record<string, string | number | boolean | null | undefined>;
    requestTimeout: number;
  };
  completion: {
    prompt: {
      experimentalStripAutoClosingCharacters: boolean;
      maxPrefixLines: number;
      maxSuffixLines: number;
    };
    debounce: {
      mode: "adaptive" | "fixed";
      interval: number;
    };
    timeout: number;
  };
  postprocess: {
    limitScope: {
      // Prefer to use syntax parser than indentation
      experimentalSyntax: boolean;
      indentation: {
        // When completion is continuing the current line, limit the scope to:
        // false(default): the line scope, meaning use the next indent level as the limit.
        // true: the block scope, meaning use the current indent level as the limit.
        experimentalKeepBlockScopeWhenCompletingLine: boolean;
      };
    };
  };
  logs: {
    level: "debug" | "error" | "silent";
  };
  anonymousUsageTracking: {
    disable: boolean;
  };
};

type RecursivePartial<T> = {
  [P in keyof T]?: T[P] extends (infer U)[]
    ? RecursivePartial<U>[]
    : T[P] extends object | undefined
    ? RecursivePartial<T[P]>
    : T[P];
};

export type PartialAgentConfig = RecursivePartial<AgentConfig>;

export const defaultAgentConfig: AgentConfig = {
  server: {
    endpoint: "http://localhost:8080",
    token: "",
    requestHeaders: {},
    requestTimeout: 30000, // 30s
  },
  completion: {
    prompt: {
      experimentalStripAutoClosingCharacters: false,
      maxPrefixLines: 20,
      maxSuffixLines: 20,
    },
    debounce: {
      mode: "adaptive",
      interval: 250, // ms
    },
    timeout: 4000, // ms
  },
  postprocess: {
    limitScope: {
      experimentalSyntax: false,
      indentation: {
        experimentalKeepBlockScopeWhenCompletingLine: false,
      },
    },
  },
  logs: {
    level: "silent",
  },
  anonymousUsageTracking: {
    disable: false,
  },
};

const configTomlTemplate = `## Tabby agent configuration file

## Online documentation: https://tabby.tabbyml.com/docs/extensions/configuration
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
## For more details on data collection, see https://tabby.tabbyml.com/docs/extensions/configuration#usage-collection
## Your contribution is greatly appreciated. However, if you prefer not to participate, you can disable anonymous usage tracking here.
# [anonymousUsageTracking]
# disable = false # set to true to disable

`;

const typeCheckSchema = {
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

function checkValueType(object, key, type) {
  if (typeof getProperty(object, key) !== type) {
    deleteProperty(object, key);
  }
}

function validateConfig(config: PartialAgentConfig): PartialAgentConfig {
  const validatedConfig = { ...config };
  for (const key in typeCheckSchema) {
    checkValueType(validatedConfig, key, typeCheckSchema[key]);
  }
  return validatedConfig;
}

export const userAgentConfig = isBrowser
  ? null
  : (() => {
      const EventEmitter = require("events");
      const fs = require("fs-extra");
      const toml = require("toml");
      const chokidar = require("chokidar");
      const deepEqual = require("deep-equal");

      class ConfigFile extends EventEmitter {
        filepath: string;
        data: PartialAgentConfig = {};
        watcher: ReturnType<typeof chokidar.watch> | null = null;
        logger = require("./logger").rootLogger.child({ component: "ConfigFile" });

        constructor(filepath: string) {
          super();
          this.filepath = filepath;
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
            if (error.code === "ENOENT") {
              await this.createTemplate();
            } else {
              this.logger.error({ error }, "Failed to load config file");
            }
          }
        }

        async createTemplate() {
          try {
            await fs.outputFile(this.filepath, configTomlTemplate);
          } catch (error) {
            this.logger.error({ error }, "Failed to create config template file");
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
      }

      const configFile = require("path").join(require("os").homedir(), ".tabby-client", "agent", "config.toml");
      return new ConfigFile(configFile);
    })();
