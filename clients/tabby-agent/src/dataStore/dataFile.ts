import { EventEmitter } from "events";
import path from "path";
import os from "os";
import fs from "fs-extra";
import chokidar from "chokidar";
import { isBrowser } from "../env";

export class FileDataStore extends EventEmitter {
  private watcher?: ReturnType<typeof chokidar.watch>;

  constructor(private readonly filepath: string) {
    super();
  }

  async read(): Promise<unknown> {
    return (await fs.readJson(this.filepath, { throws: false })) || {};
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
