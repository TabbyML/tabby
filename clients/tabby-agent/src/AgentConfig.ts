import { isBrowser } from "./env";

export type AgentConfig = {
  server: {
    endpoint: string;
    requestHeaders: Record<string, string>;
    requestTimeout: number;
  };
  completion: {
    prompt: {
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
    requestHeaders: {},
    requestTimeout: 30000, // 30s
  },
  completion: {
    prompt: {
      maxPrefixLines: 20,
      maxSuffixLines: 20,
    },
    debounce: {
      mode: "adaptive",
      interval: 250, // ms
    },
    timeout: {
      auto: 5000, // 5s
      manually: 30000, // 30s
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
## You can set the server endpoint and request timeout here.
# [server]
# endpoint = "http://localhost:8080" # http or https URL
# requestTimeout = 30000 # ms

## You can add custom request headers, e.g. for authentication.
# [server.requestHeaders]
# Authorization = "Bearer eyJhbGciOiJ..........."

## Completion
## You can set the prompt context to send to the server for completion.
# [completion.prompt]
# maxPrefixLines = 20
# maxSuffixLines = 20

## You can set the debounce mode for auto completion requests when typing.
# [completion.debounce]
# mode = "adaptive" # or "fixed"
# interval = 250 # ms, only used when mode is "fixed"

## You can set the timeout for completion requests.
# [completion.timeout]
# auto = 5000 # ms, for auto completion when typing
# manually = 30000 # ms, for manually triggered completion

## Logs
## You can set the log level here. The log file is located at ~/.tabby-client/agent/logs/.
# [logs]
# level = "silent" # or "error" or "debug"

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
            this.data = toml.parse(fileContent);
            super.emit("updated", this.data);
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
          this.watcher.on("add", this.load.bind(this));
          this.watcher.on("change", this.load.bind(this));
        }
      }

      const configFile = require("path").join(require("os").homedir(), ".tabby-client", "agent", "config.toml");
      return new ConfigFile(configFile);
    })();
