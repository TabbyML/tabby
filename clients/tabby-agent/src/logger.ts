import pino from "pino";
import { err as pinoStdSerializersError } from "pino-std-serializers";
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

const options = { serializers: { error: pinoStdSerializersError } };
export const rootLogger = !!stream ? pino(options, stream) : pino(options);
if (isTest && testLogDebug) {
  rootLogger.level = "debug";
} else {
  rootLogger.level = "silent";
}

export const allLoggers = [rootLogger];
rootLogger.onChild = (child) => {
  allLoggers.push(child);
};
