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

export const dataStore: DataStore = isBrowser
  ? null
  : (() => {
      const dataFile = require("path").join(require("os").homedir(), ".tabby-client", "agent", "data.json");
      const fs = require("fs-extra");
      return {
        data: {},
        load: async function () {
          await this.migrateFrom_0_3_0();
          this.data = (await fs.readJson(dataFile, { throws: false })) || {};
        },
        save: async function () {
          await fs.outputJson(dataFile, this.data);
        },
        migrateFrom_0_3_0: async function () {
          const dataFile_0_3_0 = require("path").join(require("os").homedir(), ".tabby", "agent", "data.json");
          const migratedFlag = require("path").join(require("os").homedir(), ".tabby", "agent", ".data_json_migrated");
          if (
            (await fs.pathExists(dataFile_0_3_0)) &&
            !(await fs.pathExists(migratedFlag))
          ) {
            const data = await fs.readJson(dataFile_0_3_0);
            await fs.outputJson(dataFile, data);
            await fs.outputFile(migratedFlag, "");
          }
        },
      };
    })();
