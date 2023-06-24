import { EventEmitter } from "events";
import decodeJwt from "jwt-decode";
import { CloudApi, DeviceTokenResponse, DeviceTokenAcceptResponse } from "./cloud";
import { ApiError, CancelablePromise } from "./generated";
import { dataStore, DataStore } from "./dataStore";
import { rootLogger } from "./logger";

export type StorageData = {
  auth: { [endpoint: string]: { jwt: string } };
};

type JWT = { token: string; payload: { email: string; exp: number } };

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
  readonly endpoint: string;
  readonly dataStore: DataStore | null = null;
  private refreshTokenTimer: ReturnType<typeof setInterval> | null = null;
  private authApi: CloudApi | null = null;
  private jwt: JWT | null = null;

  static async create(options: { endpoint: string; dataStore?: DataStore }): Promise<Auth> {
    const auth = new Auth(options);
    await auth.load();
    return auth;
  }

  constructor(options: { endpoint: string; dataStore?: DataStore }) {
    super();
    this.endpoint = options.endpoint;
    this.dataStore = options.dataStore || dataStore;
    this.authApi = new CloudApi();
    this.scheduleRefreshToken();
  }

  get token(): string | null {
    return this.jwt?.token;
  }

  get user(): string | null {
    return this.jwt?.payload.email;
  }

  private async load(): Promise<void> {
    if (!this.dataStore) return;
    try {
      await this.dataStore.load();
      const storedJwt = this.dataStore.data["auth"]?.[this.endpoint]?.jwt;
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
    } catch (error: any) {
      this.logger.debug({ error }, "Error when loading auth");
    }
  }

  private async save(): Promise<void> {
    if (!this.dataStore) return;
    try {
      if (this.jwt) {
        if (this.dataStore.data["auth"]?.[this.endpoint]?.jwt === this.jwt.token) return;
        this.dataStore.data["auth"] = { ...this.dataStore.data["auth"], [this.endpoint]: { jwt: this.jwt.token } };
      } else {
        if (typeof this.dataStore.data["auth"]?.[this.endpoint] === "undefined") return;
        delete this.dataStore.data["auth"][this.endpoint];
      }
      await this.dataStore.save();
      this.logger.debug("Save changes to data store.");
    } catch (error: any) {
      this.logger.error({ error }, "Error when saving auth");
    }
  }

  async reset(): Promise<void> {
    if (this.jwt) {
      this.jwt = null;
      await this.save();
    }
  }

  requestAuthUrl(): CancelablePromise<{ authUrl: string; code: string }> {
    return new CancelablePromise(async (resolve, reject, onCancel) => {
      let apiRequest: CancelablePromise<DeviceTokenResponse>;
      onCancel(() => {
        apiRequest?.cancel();
      });
      try {
        await this.reset();
        if (onCancel.isCancelled) return;
        this.logger.debug("Start to request device token");
        apiRequest = this.authApi.api.deviceToken({ auth_url: this.endpoint });
        const deviceToken = await apiRequest;
        this.logger.debug({ deviceToken }, "Request device token response");
        const authUrl = new URL(Auth.authPageUrl);
        authUrl.searchParams.append("code", deviceToken.data.code);
        resolve({ authUrl: authUrl.toString(), code: deviceToken.data.code });
      } catch (error) {
        this.logger.error({ error }, "Error when requesting token");
        reject(error);
      }
    });
  }

  pollingToken(code: string): CancelablePromise<boolean> {
    return new CancelablePromise((resolve, reject, onCancel) => {
      let apiRequest: CancelablePromise<DeviceTokenAcceptResponse>;
      const timer = setInterval(async () => {
        try {
          apiRequest = this.authApi.api.deviceTokenAccept({ code });
          const response = await apiRequest;
          this.logger.debug({ response }, "Poll jwt response");
          this.jwt = {
            token: response.data.jwt,
            payload: decodeJwt(response.data.jwt),
          };
          super.emit("updated", this.jwt);
          await this.save();
          clearInterval(timer);
          resolve(true);
        } catch (error) {
          if (error instanceof ApiError && [400, 401, 403, 405].indexOf(error.status) !== -1) {
            this.logger.debug({ error }, "Expected error when polling jwt");
          } else {
            // unknown error but still keep polling
            this.logger.error({ error }, "Error when polling jwt");
          }
        }
      }, Auth.tokenStrategy.polling.interval);
      setTimeout(() => {
        clearInterval(timer);
        reject(new Error("Timeout when polling token"));
      }, Auth.tokenStrategy.polling.timeout);
      onCancel(() => {
        apiRequest?.cancel();
        clearInterval(timer);
      });
    });
  }

  private async refreshToken(jwt: JWT, options = { maxTry: 1, retryDelay: 1000 }, retry = 0): Promise<JWT> {
    try {
      this.logger.debug({ retry }, "Start to refresh token");
      const refreshedJwt = await this.authApi.api.deviceTokenRefresh(jwt.token);
      this.logger.debug({ refreshedJwt }, "Refresh token response");
      return {
        token: refreshedJwt.data.jwt,
        payload: decodeJwt(refreshedJwt.data.jwt),
      };
    } catch (error) {
      if (error instanceof ApiError && [400, 401, 403, 405].indexOf(error.status) !== -1) {
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
      throw { ...error, retry };
    }
  }

  private scheduleRefreshToken() {
    this.refreshTokenTimer = setInterval(async () => {
      if (!this.jwt) {
        return null;
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
