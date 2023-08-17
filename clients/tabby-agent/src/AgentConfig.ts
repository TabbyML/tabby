import { isBrowser } from "./env";

export type AgentConfig = {
  server: {
    endpoint: string;
    requestHeaders: Record<string, string>;
    requestTimeout: number;
  };
  completion: {
    timeout: {
      auto: number;
      manually: number;
    };
    maxPrefixLines: number;
    maxSuffixLines: number;
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
    timeout: {
      auto: 5000, // 5s
      manually: 30000, // 30s
    },
    maxPrefixLines: 20,
    maxSuffixLines: 20,
  },
  logs: {
    level: "silent",
  },
  anonymousUsageTracking: {
    disable: false,
  },
};

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
            this.logger.error({ error }, "Failed to load config file");
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

      const configFile = require("path").join(require("os").homedir(), ".tabby", "agent", "config.toml");
      return new ConfigFile(configFile);
    })();
