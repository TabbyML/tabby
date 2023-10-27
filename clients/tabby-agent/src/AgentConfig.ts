import semverCompare from "semver-compare";
import { isBrowser } from "./env";

const configVersion = "1.1.0";

export type AgentConfig = {
  version: string;
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
    limitScopeByIndentation: {
      // When completion is continuing the current line, limit the scope to:
      // false(default): the line scope, meaning use the next indent level as the limit.
      // true: the block scope, meaning use the current indent level as the limit.
      experimentalKeepBlockScopeWhenCompletingLine: boolean;
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
  version: configVersion,
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
    limitScopeByIndentation: {
      experimentalKeepBlockScopeWhenCompletingLine: false,
    },
  },
  logs: {
    level: "silent",
  },
  anonymousUsageTracking: {
    disable: false,
  },
};

function generateConfigFile(isMigrated: boolean, config: PartialAgentConfig) {
  let file = `## Tabby agent configuration file\n`;
  file += `version = "${configVersion}" # Version of the config file, managed by Tabby. Do not edit.\n`;
  file += "\n";

  if (isMigrated) {
    file += `## This file is automatically updated by Tabby.\n`;
    file += `## You configuration is migrated and the old config file is retained as ~/.tabby-client/agent/config.toml.{date}.\n`;
    file += "\n";
  }
  file += `## You can edit this file to customize your settings.\n`;
  file += `## Note that configurations in this file has lower priority than in IDE settings.\n`;
  file += "\n";

  file += `## Server\n`;
  file += `## You can set the server endpoint and authentication token here.\n`;
  if (
    config.server?.endpoint !== undefined ||
    config.server?.token !== undefined ||
    config.server?.requestHeaders?.["Authorization"] !== undefined
  ) {
    file += `[server]\n`;
    if (config.server?.endpoint !== undefined) {
      file += `endpoint = "${config.server.endpoint}" # http or https URL\n`;
    } else {
      file += `# endpoint = "http://localhost:8080" # http or https URL\n`;
    }
    if (config.server?.token !== undefined || config.server?.requestHeaders?.["Authorization"] !== undefined) {
      const token =
        config.server?.token ?? config.server?.requestHeaders["Authorization"]?.toString()?.replace("Bearer ", "");
      file += `token = "${token}" # if server requires authentication\n`;
    } else {
      file += `# token = "your-token-here" # if server requires authentication\n`;
    }
  } else {
    file += `# [server]\n`;
    file += `# endpoint = "http://localhost:8080" # http or https URL\n`;
    file += `# token = "your-token-here" # if server requires authentication\n`;
  }
  file += "\n";

  file += `## Logs\n`;
  file += `## You can set the log level here.\n`;
  file += `## The log file is located at ~/.tabby-client/agent/logs/.\n`;

  if (config.logs?.level !== undefined) {
    file += `[logs]\n`;
    file += `level = "${config.logs.level}" # "silent" or "error" or "debug"\n`;
  } else {
    file += `# [logs]\n`;
    file += `# level = "silent" # "silent" or "error" or "debug"\n`;
  }
  file += "\n";

  file += `## Anonymous usage tracking\n`;
  file += `## You can disable anonymous usage tracking here.\n`;
  if (config.anonymousUsageTracking?.disable !== undefined) {
    file += `[anonymousUsageTracking]\n`;
    file += `disable = ${config.anonymousUsageTracking.disable} # set to true to disable\n`;
  } else {
    file += `# [anonymousUsageTracking]\n`;
    file += `# disable = false # set to true to disable\n`;
  }
  file += "\n";

  return file;
}

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
            const data = toml.parse(fileContent);
            if (semverCompare(data.version ?? "0", configVersion) < 0) {
              await this.migrateConfig(data);
              return await this.load();
            }
            this.data = data;
            super.emit("updated", this.data);
          } catch (error) {
            if (error.code === "ENOENT") {
              await this.createTemplate();
            } else {
              this.logger.error({ error }, "Failed to load config file");
            }
          }
        }

        async migrateConfig(config) {
          try {
            const suffix = `.${new Date().toISOString().replace(/\D+/g, "")}`;
            await fs.move(this.filepath, `${this.filepath}${suffix}`);
            await fs.outputFile(this.filepath, generateConfigFile(true, config));
            this.logger.info(`Migrated config file to ${this.filepath}${suffix}.`);
          } catch (error) {
            this.logger.error({ error }, "Failed to migrate config file");
          }
        }

        async createTemplate() {
          try {
            await fs.outputFile(this.filepath, generateConfigFile(false, {}));
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
