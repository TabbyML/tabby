import { isBrowser } from "./env";

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
    timeout: {
      auto: number;
      manually: number;
    };
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
    // Deprecated: There is a timeout of 3s on the server side since v0.3.0.
    timeout: {
      auto: 4000, // 4s
      manually: 4000, // 4s
    },
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

## You can uncomment any block to enable settings.
## Configurations in this file has lower priority than in IDE settings.

## Server
## You can set the server endpoint here, and auth token if server requires.
# [server]
# endpoint = "http://localhost:8080" # http or https URL
# token = "your-token-here" # if token is set, request header Authorization = "Bearer $token" will be added automatically

## You can add custom request headers.
# [server.requestHeaders]
# Header1 = "Value1" # list your custom headers here
# Header2 = "Value2" # value can be string, number or boolean

## Logs
## You can set the log level here. The log file is located at ~/.tabby-client/agent/logs/.
# [logs]
# level = "silent" # "silent" or "error" or "debug"

## Anonymous usage tracking
## You can disable anonymous usage tracking here.
# [anonymousUsageTracking]
# disable = false # set to true to disable

`;

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
            this.data = data;
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
