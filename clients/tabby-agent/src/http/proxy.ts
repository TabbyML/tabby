import { Dispatcher, ProxyAgent, EnvHttpProxyAgent } from "undici";
import { parse as uriParse } from "uri-js";
import { isBrowser } from "../env";
import { getLogger } from "../logger";

export type ProxyConfig =
  | {
      url: string;
      authorization?: string;
      noProxy?: string;
    }
  | { fromEnv: true };

const logger = getLogger("Proxy");

export function createProxyForUrl(url: string, configs: ProxyConfig[]): Dispatcher | null {
  if (isBrowser) {
    return null;
  }
  const { host } = uriParse(url);
  if (!host) {
    return null;
  }
  for (const config of configs) {
    if ("fromEnv" in config && config["fromEnv"]) {
      logger.info("Using proxy from environment variables.");
      return new EnvHttpProxyAgent();
    } else if ("url" in config && config["url"]) {
      const noProxyList = config["noProxy"]?.split(/,|\s+/).map((item) => item.trim());
      if (!noProxyList || !noProxyList.includes(host)) {
        logger.info(`Using proxy ${config["url"]}.`);
        return new ProxyAgent({
          uri: config["url"],
          token: config["authorization"],
        });
      }
    }
  }
  return null;
}
