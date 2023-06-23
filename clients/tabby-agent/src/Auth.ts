import { EventEmitter } from "events";
import decodeJwt from "jwt-decode";
import { CloudApi } from "./cloud";
import { ApiError } from "./generated";
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
      // refresh token 30 min before token expires
      // assume a new token expires in 1 day, much longer than 30 min
      beforeExpire: 30 * 60 * 1000,
      maxTry: 5, // try to refresh token 5 times
      retryDelay: 2000, // retry after 2 seconds
    },
  };

  private readonly logger = rootLogger.child({ component: "Auth" });
  readonly endpoint: string;
  readonly dataStore: DataStore | null = null;
  private pollingTokenTimer: ReturnType<typeof setInterval> | null = null;
  private stopPollingTokenTimer: ReturnType<typeof setTimeout> | null = null;
  private refreshTokenTimer: ReturnType<typeof setTimeout> | null = null;
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

    // From tabby endpoint: http[s]://{namespace}.app.tabbyml.com/tabby[/]
    // To auth endpoint: http[s]://{namespace}.app.tabbyml.com/api
    const authApiBase = this.endpoint.replace(/\/tabby\/?$/, "/api");
    this.authApi = new CloudApi({ BASE: authApiBase });
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
          this.jwt = await this.refreshToken(jwt);
          await this.save();
        } else {
          this.jwt = jwt;
        }
        this.scheduleRefreshToken();
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
    if (this.refreshTokenTimer) {
      clearTimeout(this.refreshTokenTimer);
      this.refreshTokenTimer = null;
    }
    if (this.pollingTokenTimer) {
      clearInterval(this.pollingTokenTimer);
      this.pollingTokenTimer = null;
    }
    if (this.stopPollingTokenTimer) {
      clearTimeout(this.stopPollingTokenTimer);
      this.stopPollingTokenTimer = null;
    }
  }

  async requestToken(): Promise<string> {
    try {
      await this.reset();
      this.logger.debug("Start to request device token");
      const deviceToken = await this.authApi.api.deviceToken({ auth_url: this.endpoint });
      this.logger.debug({ deviceToken }, "Request device token response");
      const authUrl = new URL(Auth.authPageUrl);
      authUrl.searchParams.append("code", deviceToken.data.code);
      this.schedulePollingToken(deviceToken.data.code);
      return authUrl.toString();
    } catch (error) {
      this.logger.error({ error }, "Error when requesting token");
      throw error;
    }
  }

  private async refreshToken(jwt: JWT, retry = 0): Promise<JWT> {
    try {
      this.logger.debug({ retry }, "Start to refresh token");
      const refreshedJwt = await this.authApi.api.deviceTokenRefresh(jwt.token);
      this.logger.debug({ refreshedJwt }, "Refresh token response");
      return {
        token: refreshedJwt.data.jwt,
        payload: decodeJwt(refreshedJwt.data.jwt),
      };
    } catch (error) {
      if (error instanceof ApiError && [401, 403, 405].indexOf(error.status) !== -1) {
        this.logger.debug({ error }, "Error when refreshing jwt");
      } else {
        // unknown error, retry a few times
        this.logger.error({ error }, "Unknown error when refreshing jwt");
        if (retry < Auth.tokenStrategy.refresh.maxTry) {
          await new Promise((resolve) => setTimeout(resolve, Auth.tokenStrategy.refresh.retryDelay));
          this.logger.debug("Retry refreshing jwt");
          return this.refreshToken(jwt, retry + 1);
        }
      }
      throw { ...error, retry };
    }
  }

  private async schedulePollingToken(code: string) {
    this.pollingTokenTimer = setInterval(async () => {
      try {
        const response = await this.authApi.api.deviceTokenAccept({ code });
        this.logger.debug({ response }, "Poll jwt response");
        this.jwt = {
          token: response.data.jwt,
          payload: decodeJwt(response.data.jwt),
        };
        await this.save();
        this.scheduleRefreshToken();
        super.emit("updated", this.jwt);
        clearInterval(this.pollingTokenTimer);
        this.pollingTokenTimer = null;
      } catch (error) {
        if (error instanceof ApiError && [401, 403, 405].indexOf(error.status) !== -1) {
          this.logger.debug({ error }, "Expected error when polling jwt");
        } else {
          // unknown error but still keep polling
          this.logger.error({ error }, "Error when polling jwt");
        }
      }
    }, Auth.tokenStrategy.polling.interval);
    this.stopPollingTokenTimer = setTimeout(() => {
      if (this.pollingTokenTimer) {
        clearInterval(this.pollingTokenTimer);
        this.pollingTokenTimer = null;
      }
    }, Auth.tokenStrategy.polling.timeout);
  }

  private scheduleRefreshToken() {
    if (this.refreshTokenTimer) {
      clearTimeout(this.refreshTokenTimer);
      this.refreshTokenTimer = null;
    }
    if (!this.jwt) {
      return null;
    }

    const refreshDelay = Math.max(
      0,
      this.jwt.payload.exp * 1000 - Auth.tokenStrategy.refresh.beforeExpire - Date.now()
    );
    this.logger.debug({ refreshDelay }, "Schedule refresh token");
    this.refreshTokenTimer = setTimeout(async () => {
      this.jwt = await this.refreshToken(this.jwt);
      await this.save();
      this.scheduleRefreshToken();
      super.emit("updated", this.jwt);
    }, refreshDelay);
  }
}
