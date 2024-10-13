import { EventEmitter } from "events";
import path from "path";
import os from "os";
import fs from "fs-extra";
import chokidar from "chokidar";
import { isBrowser } from "../env";
import { getLogger } from "../logger";

export class FileDataStore extends EventEmitter {
  private readonly logger = getLogger("DataStore");
  private watcher?: ReturnType<typeof chokidar.watch>;

  constructor(private readonly filepath: string) {
    super();
  }

  async read(): Promise<unknown> {
    try {
      const json = await fs.readJson(this.filepath, { throws: false });
      return json ?? {};
    } catch (err) {
      this.logger.warn(`Failed to read ${this.filepath}: ${err}`);
      return {};
    }
  }

  async write(data: unknown) {
    await fs.outputJson(this.filepath, data);
  }

  watch() {
    this.watcher = chokidar.watch(this.filepath, {
      interval: 1000,
    });
    const onUpdated = async () => {
      this.emit("updated");
    };
    this.watcher.on("add", onUpdated);
    this.watcher.on("change", onUpdated);
  }
}

export function getFileDataStore(): FileDataStore | undefined {
  const dataFilePath = path.join(os.homedir(), ".tabby-client", "agent", "data.json");
  return isBrowser ? undefined : new FileDataStore(dataFilePath);
}
