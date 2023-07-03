import pino from "pino";
import { isBrowser } from "./env";

/**
 * Stream not available in browser, will use default console output.
 */
const stream = isBrowser
  ? null
  : /**
     * Default rotating file locate at `~/.tabby/agent/logs/`.
     */
    require("rotating-file-stream").createStream("tabby-agent.log", {
      path: require("path").join(require("os").homedir(), ".tabby", "agent", "logs"),
      size: "10M",
      interval: "1d",
    });

export const rootLogger = !!stream ? pino(stream) : pino();

export const allLoggers = [rootLogger];
rootLogger.onChild = (child) => {
  allLoggers.push(child);
};
