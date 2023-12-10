import { EventEmitter } from "events";
import path from "path";
import os from "os";
import fs from "fs-extra";
import deepEqual from "deep-equal";
import chokidar from "chokidar";
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

class FileDataStore extends EventEmitter implements FileDataStore {
  private watcher?: ReturnType<typeof chokidar.watch>;
  public data: Partial<StoredData> = {};

  constructor(private readonly filepath: string) {
    super();
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
    };
    this.watcher.on("add", onChanged);
    this.watcher.on("change", onChanged);
  }
}

const dataFile = path.join(os.homedir(), ".tabby-client", "agent", "data.json");
export const dataStore = isBrowser ? undefined : new FileDataStore(dataFile);
