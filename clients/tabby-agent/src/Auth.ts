import { EventEmitter } from "events";
import decodeJwt from "jwt-decode";
import createClient from "openapi-fetch";
import type { paths as CloudApi } from "./types/cloudApi";
import type { AbortSignalOption } from "./Agent";
import { HttpError, abortSignalFromAnyOf } from "./utils";
import { dataStore, DataStore } from "./dataStore";
import { rootLogger } from "./logger";

export type StorageData = {
  auth: { [endpoint: string]: { jwt: string } };
};

type JWT = { token: string; payload: { email: string; exp: number } };

class RetryLimitReachedError extends Error {
  readonly name = "RetryLimitReachedError";
  constructor(readonly cause: unknown) {
    super();
  }
}

export class Auth extends EventEmitter {
  static readonly authPageUrl = "https://app.tabbyml.com/account/device-token";
  static readonly tokenStrategy = {
    polling: {
      // polling token after auth url generated
      interval: 5000, // polling token every 5 seconds
      timeout: 5 * 60 * 1000, // stop polling after trying for 5 min
    },
    refresh: {
      // check token every 15 min, refresh token if it expires in 30 min
      interval: 15 * 60 * 1000,
      beforeExpire: 30 * 60 * 1000,
      whenLoaded: {
        // after token loaded from data store, refresh token if it is about to expire or has expired
        maxTry: 5, // keep loading time not too long
        retryDelay: 1000, // retry after 1 seconds
      },
      whenScheduled: {
        // if running until token is about to expire, refresh token as scheduled
        maxTry: 60,
        retryDelay: 30 * 1000, // retry after 30 seconds
      },
    },
  };

  private readonly logger = rootLogger.child({ component: "Auth" });
  private dataStore?: DataStore;
  private authApi = createClient<CloudApi>({ baseUrl: "https://app.tabbyml.com/api" });
  private jwt?: JWT;

  constructor(readonly endpoint: string) {
    super();
  }

  async init(options?: { dataStore?: DataStore }) {
    if (options?.dataStore) {
      this.dataStore = options.dataStore;
    } else {
      this.dataStore = dataStore;
      if (dataStore) {
        dataStore.on("updated", async () => {
          await this.load();
          super.emit("updated", this.jwt);
        });
        dataStore.watch();
      }
    }
    this.scheduleRefreshToken();
    await this.load();
  }

  get token(): string | undefined {
    return this.jwt?.token;
  }

  get user(): string | undefined {
    return this.jwt?.payload.email;
  }

  private async load(): Promise<void> {
    if (!this.dataStore) {
      return;
    }
    try {
      await this.dataStore.load();
      const storedJwt = this.dataStore.data.auth?.[this.endpoint]?.jwt;
      if (typeof storedJwt === "string" && this.jwt?.token !== storedJwt) {
        this.logger.debug({ storedJwt }, "Load jwt from data store.");
        const jwt: JWT = {
          token: storedJwt,
          payload: decodeJwt(storedJwt),
        };
        // refresh token if it is about to expire or has expired
        if (jwt.payload.exp * 1000 - Date.now() < Auth.tokenStrategy.refresh.beforeExpire) {
          this.jwt = await this.refreshToken(jwt, Auth.tokenStrategy.refresh.whenLoaded);
          await this.save();
        } else {
          this.jwt = jwt;
        }
      }
    } catch (error) {
      this.logger.debug({ error }, "Error when loading auth");
    }
  }

  private async save(): Promise<void> {
    if (!this.dataStore) {
      return;
    }
    try {
      if (this.jwt) {
        if (this.dataStore.data.auth?.[this.endpoint]?.jwt === this.jwt.token) {
          return;
        }
        this.dataStore.data.auth = { ...this.dataStore.data.auth, [this.endpoint]: { jwt: this.jwt.token } };
      } else {
        if (typeof this.dataStore.data.auth?.[this.endpoint] === "undefined") {
          return;
        }
        delete this.dataStore.data.auth[this.endpoint];
      }
      await this.dataStore.save();
      this.logger.debug("Save changes to data store.");
    } catch (error) {
      this.logger.error({ error }, "Error when saving auth");
    }
  }

  async reset(): Promise<void> {
    if (this.jwt) {
      this.jwt = undefined;
      await this.save();
    }
  }

  async requestAuthUrl(options?: AbortSignalOption): Promise<{ authUrl: string; code: string }> {
    try {
      await this.reset();
      if (options?.signal.aborted) {
        throw options.signal.reason;
      }
      this.logger.debug("Start to request device token");
      const response = await this.authApi.POST("/device-token", {
        body: { auth_url: this.endpoint },
        signal: options?.signal,
      });
      if (response.error || !response.response.ok) {
        throw new HttpError(response.response);
      }
      const deviceToken = response.data;
      this.logger.debug({ deviceToken }, "Request device token response");
      const authUrl = new URL(Auth.authPageUrl);
      authUrl.searchParams.append("code", deviceToken.data.code);
      return { authUrl: authUrl.toString(), code: deviceToken.data.code };
    } catch (error) {
      this.logger.error({ error }, "Error when requesting token");
      throw error;
    }
  }

  async pollingToken(code: string, options?: AbortSignalOption): Promise<boolean> {
    return new Promise((resolve, reject) => {
      const signal = abortSignalFromAnyOf([AbortSignal.timeout(Auth.tokenStrategy.polling.timeout), options?.signal]);
      const timer = setInterval(async () => {
        try {
          const response = await this.authApi.POST("/device-token/accept", { params: { query: { code } }, signal });
          if (response.error || !response.response.ok) {
            throw new HttpError(response.response);
          }
          const result = response.data;
          this.logger.debug({ result }, "Poll jwt response");
          this.jwt = {
            token: result.data.jwt,
            payload: decodeJwt(result.data.jwt),
          };
          super.emit("updated", this.jwt);
          await this.save();
          clearInterval(timer);
          resolve(true);
        } catch (error) {
          if (error instanceof HttpError && [400, 401, 403, 405].includes(error.status)) {
            this.logger.debug({ error }, "Expected error when polling jwt");
          } else {
            // unknown error but still keep polling
            this.logger.error({ error }, "Error when polling jwt");
          }
        }
      }, Auth.tokenStrategy.polling.interval);
      if (signal.aborted) {
        clearInterval(timer);
        reject(signal.reason);
      } else {
        signal.addEventListener("abort", () => {
          clearInterval(timer);
          reject(signal.reason);
        });
      }
    });
  }

  private async refreshToken(jwt: JWT, options = { maxTry: 1, retryDelay: 1000 }, retry = 0): Promise<JWT> {
    try {
      this.logger.debug({ retry }, "Start to refresh token");
      const response = await this.authApi.POST("/device-token/refresh", {
        headers: { Authorization: `Bearer ${jwt.token}` },
      });
      if (response.error || !response.response.ok) {
        throw new HttpError(response.response);
      }
      const refreshedJwt = response.data;
      this.logger.debug({ refreshedJwt }, "Refresh token response");
      return {
        token: refreshedJwt.data.jwt,
        payload: decodeJwt(refreshedJwt.data.jwt),
      };
    } catch (error) {
      if (error instanceof HttpError && [400, 401, 403, 405].includes(error.status)) {
        this.logger.debug({ error }, "Error when refreshing jwt");
      } else {
        // unknown error, retry a few times
        this.logger.error({ error }, "Unknown error when refreshing jwt");
        if (retry < options.maxTry) {
          this.logger.debug(`Retry refreshing jwt after ${options.retryDelay}ms`);
          await new Promise((resolve) => setTimeout(resolve, options.retryDelay));
          return this.refreshToken(jwt, options, retry + 1);
        }
      }
      throw new RetryLimitReachedError(error);
    }
  }

  private scheduleRefreshToken() {
    setInterval(async () => {
      if (!this.jwt) {
        return;
      }
      if (this.jwt.payload.exp * 1000 - Date.now() < Auth.tokenStrategy.refresh.beforeExpire) {
        try {
          this.jwt = await this.refreshToken(this.jwt, Auth.tokenStrategy.refresh.whenScheduled);
          super.emit("updated", this.jwt);
          await this.save();
        } catch (error) {
          this.logger.error({ error }, "Error when refreshing jwt");
        }
      } else {
        this.logger.debug("Check token, still valid");
      }
    }, Auth.tokenStrategy.refresh.interval);
  }
}
