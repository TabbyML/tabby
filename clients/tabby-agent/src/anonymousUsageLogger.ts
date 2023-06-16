import { CloudApi } from "./cloud";
import { rootLogger } from "./logger";
import { isBrowser } from "./utils";
import { name as agentName, version as agentVersion } from "../package.json";

const anonymousUsageTrackingApi = new CloudApi({ BASE: "https://app.tabbyml.com/api" });
const logger = rootLogger.child({ component: "AnonymousUsage" });
const systemData = {
  agent: { name: agentName, version: agentVersion },
  browser: isBrowser ? navigator?.userAgent || "browser" : undefined,
  node: isBrowser ? undefined : `${process.version} ${process.platform} ${require("os").arch()} ${require("os").release()}`,
};


export const anonymousUsageLogger = {
  event: async (data: any) => {
    await anonymousUsageTrackingApi.api
      .usage({
        ...systemData,
        ...data,
      })
      .catch((error) => {
        logger.error({ error }, "Error when sending anonymous usage data");
      });
  },
};
