import { isBrowser } from "./env";

export type StoredData = {
  anonymousId: string;
  auth: { [endpoint: string]: { jwt: string } };
};

export interface DataStore {
  data: Partial<StoredData>;
  load(): PromiseLike<void>;
  save(): PromiseLike<void>;
}

export const dataStore = isBrowser
  ? null
  : (() => {
      const EventEmitter = require("events");
      const fs = require("fs-extra");
      const deepEqual = require("deep-equal");
      const chokidar = require("chokidar");

      class FileDataStore extends EventEmitter implements FileDataStore {
        filepath: string;
        data: Partial<StoredData> = {};
        watcher: ReturnType<typeof chokidar.watch> | null = null;

        constructor(filepath: string) {
          super();
          this.filepath = filepath;
        }

        async load() {
          this.data = (await fs.readJson(dataFile, { throws: false })) || {};
        }

        async save() {
          await fs.outputJson(dataFile, this.data);
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
          }
          this.watcher.on("add", onChanged);
          this.watcher.on("change", onChanged);
        }
      }

      const dataFile = require("path").join(require("os").homedir(), ".tabby-client", "agent", "data.json");
      return new FileDataStore(dataFile);
    })();
