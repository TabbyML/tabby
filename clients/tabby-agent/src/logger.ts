import path from "path";
import os from "os";
import pino from "pino";
import { createStream } from "rotating-file-stream";
import { isBrowser, isTest, testLogDebug } from "./env";

/**
 * Stream not available in browser, will use default console output.
 */
const stream =
  isBrowser || isTest
    ? undefined
    : /**
       * Default rotating file locate at `~/.tabby-client/agent/logs/`.
       */
      createStream("tabby-agent.log", {
        path: path.join(os.homedir(), ".tabby-client", "agent", "logs"),
        size: "10M",
        interval: "1d",
      });

const options = { serializers: { error: pino.stdSerializers.err } };
export const rootLogger = stream ? pino(options, stream) : pino(options);
if (isTest && testLogDebug) {
  rootLogger.level = "debug";
} else {
  rootLogger.level = "silent";
}

export const allLoggers = [rootLogger];
rootLogger.onChild = (child: pino.Logger) => {
  allLoggers.push(child);
};
