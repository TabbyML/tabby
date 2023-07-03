import { isBrowser } from "./env";

export type AgentConfig = {
  server: {
    endpoint: string;
  };
  logs: {
    level: "debug" | "error" | "silent";
  };
  anonymousUsageTracking: {
    disable: boolean;
  };
};

export const defaultAgentConfig: AgentConfig = {
  server: {
    endpoint: "http://localhost:8080",
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
        data: Partial<AgentConfig> = {};
        watcher: ReturnType<typeof chokidar.watch> | null = null;
        logger = require("./logger").rootLogger.child({ component: "ConfigFile" });

        constructor(filepath: string) {
          super();
          this.filepath = filepath;
        }

        get config(): Partial<AgentConfig> {
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
