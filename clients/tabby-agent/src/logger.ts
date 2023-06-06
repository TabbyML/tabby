import os from "os";
import path from "path";
import { isBrowser } from "browser-or-node";
import * as rotatingFileStream from "rotating-file-stream";
import pino from "pino";

/**
 * Stream not available in browser, will use default console output.
 */
const stream = isBrowser
  ? null
  : /**
     * Default rotating file locate at `~/.tabby/agent-logs/`.
     */
    rotatingFileStream.createStream("tabby-agent.log", {
      path: path.join(os.homedir(), ".tabby", "agent-logs"),
      size: "10M",
      interval: "1d",
    });

export const rootLogger = !!stream ? pino(stream) : pino();

export const allLoggers = [rootLogger];
rootLogger.onChild = (child) => {
  allLoggers.push(child);
};
