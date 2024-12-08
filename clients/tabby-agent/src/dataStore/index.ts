import type { Connection } from "vscode-languageserver";
import type {
  ClientCapabilities,
  ClientProvidedConfig,
  DataStoreRecords,
  ServerCapabilities,
  StatusIssuesName,
} from "../protocol";
import type { TabbyServerProvidedConfig } from "../http/tabbyApiClient";
import type { Feature } from "../feature";
import type { FileDataStore } from "./dataFile";
import { EventEmitter } from "events";
import { DataStoreDidUpdateNotification, DataStoreUpdateRequest } from "../protocol";
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
  private lspInitialized = false;

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
          this.emit("updated", this.data, old);
        }
      });
      dataStore.watch();
    }
  }

  async initialize(
    connection: Connection,
    clientCapabilities: ClientCapabilities,
    _clientProvidedConfig: ClientProvidedConfig,
    dataStoreRecords: DataStoreRecords | undefined,
  ): Promise<ServerCapabilities> {
    if (clientCapabilities.tabby?.dataStore) {
      this.lspConnection = connection;

      // When dataStore is provided by the LSP connection, do not use the file data store anymore.
      const dataStore = this.fileDataStore;
      if (dataStore) {
        dataStore.stopWatch();
        this.fileDataStore = undefined;
        const old = this.data;
        this.data = dataStoreRecords ?? {};
        this.emit("updated", this.data, old);
      } else {
        this.data = dataStoreRecords ?? {};
      }

      connection.onNotification(DataStoreDidUpdateNotification.type, async (params) => {
        const records = params ?? {};
        if (!deepEqual(records, this.data)) {
          const old = this.data;
          this.data = records;
          this.emit("updated", this.data, old);
        }
      });
    }
    return {};
  }

  async initialized() {
    if (this.lspConnection) {
      this.lspInitialized = true;
      this.emit("initialized");
    }
  }

  async save() {
    if (this.lspConnection) {
      const connection = this.lspConnection;
      const sendUpdateRequest = async () => {
        await connection.sendRequest(DataStoreUpdateRequest.type, this.data);
      };
      if (this.lspInitialized) {
        await sendUpdateRequest();
      } else {
        this.once("initialized", async () => {
          await sendUpdateRequest();
        });
      }
    } else if (this.fileDataStore) {
      const oldData = (await this.fileDataStore.read()) as Partial<StoredData>;
      if (!deepEqual(oldData, this.data)) {
        await this.fileDataStore.write(this.data);
        this.emit("updated", this.data, oldData);
      }
    }
  }
}
