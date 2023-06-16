import { EventEmitter } from "events";
import { CloudApi } from "./cloud";
import { ApiError } from "./generated";
import { dataStore, DataStore } from "./dataStore";
import { rootLogger } from "./logger";

export type StorageData = {
  auth: { [endpoint: string]: { jwt: string } };
};

export class Auth extends EventEmitter {
  static readonly authPageUrl = "https://app.tabbyml.com/account/device-token";
  static readonly pollTokenInterval = 5000; // 5 seconds
  static readonly refreshTokenInterval = 1000 * 60 * 60 * 24 * 3; // 3 days

  private readonly logger = rootLogger.child({ component: "Auth" });
  readonly endpoint: string;
  readonly dataStore: DataStore | null = null;
  private pollingTokenTimer: ReturnType<typeof setInterval> | null = null;
  private refreshTokenTimer: ReturnType<typeof setTimeout> | null = null;
  private authApi: CloudApi | null = null;
  private jwt: string | null = null;

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
    return this.jwt;
  }

  private async load(): Promise<void> {
    if (!this.dataStore) return;
    try {
      await this.dataStore.load();
      const storedJwt = this.dataStore.data["auth"]?.[this.endpoint]?.jwt;
      if (typeof storedJwt === "string" && this.jwt !== storedJwt) {
        this.logger.debug({ storedJwt }, "Load jwt from data store.");
        this.jwt = storedJwt;
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
        if (this.dataStore.data["auth"]?.[this.endpoint]?.jwt === this.jwt) return;
        this.dataStore.data["auth"] = { ...this.dataStore.data["auth"], [this.endpoint]: { jwt: this.jwt } };
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
  }

  async requestToken(): Promise<string> {
    try {
      await this.reset();
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

  async schedulePollingToken(code: string) {
    this.pollingTokenTimer = setInterval(async () => {
      try {
        const response = await this.authApi.api.deviceTokenAccept({ code });
        this.logger.debug({ response }, "Poll jwt response");
        this.jwt = response.data.jwt;
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
    }, Auth.pollTokenInterval);
  }

  private scheduleRefreshToken() {
    if (this.refreshTokenTimer) {
      clearTimeout(this.refreshTokenTimer);
      this.refreshTokenTimer = null;
    }
    if (!this.jwt) {
      return null;
    }
    // FIXME: assume jwt expires after 7 days, should get exp from decode jwt payload
    const expireAt = Date.now() / 1000 + 60 * 60 * 24 * 7;
    const refreshDelay = Math.max(0, expireAt * 1000 - Date.now() - Auth.refreshTokenInterval);
    this.refreshTokenTimer = setTimeout(async () => {
      this.logger.debug({ expireAt }, "Refresh token");
      // FIXME: not implemented
    }, refreshDelay);
  }
}
