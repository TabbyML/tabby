import type { Connection } from "vscode-languageserver";
import type {
  ClientCapabilities,
  ServerCapabilities,
  StatusIssuesName,
  DataStoreGetParams,
  DataStoreSetParams,
} from "../protocol";
import type { TabbyServerProvidedConfig } from "../http/tabbyApiClient";
import type { Feature } from "../feature";
import type { FileDataStore } from "./dataFile";
import { EventEmitter } from "events";
import { DataStoreGetRequest, DataStoreSetRequest } from "../protocol";
import { getFileDataStore } from "./dataFile";
import deepEqual from "deep-equal";

export type StoredData = {
  anonymousId: string;
  serverConfig: { [endpoint: string]: TabbyServerProvidedConfig };
  statusIgnoredIssues: StatusIssuesName[];
};

export class DataStore extends EventEmitter implements Feature {
  public data: Partial<StoredData> = {};

  private lspConnection: Connection | undefined = undefined;
  private fileDataStore: FileDataStore | undefined = undefined;

  async preInitialize(): Promise<void> {
    const dataStore = getFileDataStore();
    if (dataStore) {
      this.fileDataStore = dataStore;

      this.data = (await dataStore.read()) as Partial<StoredData>;
      dataStore.on("updated", async () => {
        const data = (await dataStore.read()) as Partial<StoredData>;
        if (!deepEqual(data, this.data)) {
          const old = this.data;
          this.data = data;
          this.emit("updated", data, old);
        }
      });
      dataStore.watch();
    }
  }

  async initialize(connection: Connection, clientCapabilities: ClientCapabilities): Promise<ServerCapabilities> {
    if (clientCapabilities.tabby?.dataStore) {
      this.lspConnection = connection;

      const params: DataStoreGetParams = { key: "data" };
      const data = await connection.sendRequest(DataStoreGetRequest.type, params);
      if (!deepEqual(data, this.data)) {
        const old = this.data;
        this.data = data;
        this.emit("updated", data, old);
      }
    }
    return {};
  }

  async save() {
    if (this.lspConnection) {
      const params: DataStoreGetParams = { key: "data" };
      const old = await this.lspConnection.sendRequest(DataStoreGetRequest.type, params);

      if (!deepEqual(old, this.data)) {
        const params: DataStoreSetParams = { key: "data", value: this.data };
        await this.lspConnection.sendRequest(DataStoreSetRequest.type, params);
        this.emit("updated", this.data, old);
      }
    }

    if (this.fileDataStore) {
      await this.fileDataStore.write(this.data);
    }
  }
}
