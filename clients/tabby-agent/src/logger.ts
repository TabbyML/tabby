import pino from "pino";
import { isBrowser, isTest, testLogDebug } from "./env";

/**
 * Stream not available in browser, will use default console output.
 */
const stream =
  isBrowser || isTest
    ? null
    : /**
       * Default rotating file locate at `~/.tabby-client/agent/logs/`.
       */
      require("rotating-file-stream").createStream("tabby-agent.log", {
        path: require("path").join(require("os").homedir(), ".tabby-client", "agent", "logs"),
        size: "10M",
        interval: "1d",
      });

export const rootLogger = !!stream ? pino(stream) : pino();
if (isTest && testLogDebug) {
  rootLogger.level = "debug";
} else {
  rootLogger.level = "silent";
}

export const allLoggers = [rootLogger];
rootLogger.onChild = (child) => {
  allLoggers.push(child);
};
