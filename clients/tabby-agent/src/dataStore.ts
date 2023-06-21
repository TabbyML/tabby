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
      const dataFile = require("path").join(require("os").homedir(), ".tabby", "agent", "data.json");
      const fs = require("fs-extra");
      return {
        data: {},
        load: async function () {
          this.data = (await fs.readJson(dataFile, { throws: false })) || {};
        },
        save: async function () {
          await fs.outputJson(dataFile, this.data);
        },
      };
    })();
