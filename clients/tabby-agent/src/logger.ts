import path from "path";
import os from "os";
import pino from "pino";
import * as FileStreamRotator from "file-stream-rotator";
import { isBrowser, isTest, testLogDebug } from "./env";

class LogFileStream implements pino.DestinationStream {
  private streamOptions = {
    // Rotating file locate at `~/.tabby-client/agent/logs/`.
    filename: path.join(os.homedir(), ".tabby-client", "agent", "logs", "%DATE%"),
    frequency: "daily",
    size: "10M",
    max_logs: "30d",
    extension: ".log",
    create_symlink: true,
  };
  private stream?: pino.DestinationStream;

  write(data: string): void {
    if (!this.stream) {
      this.stream = FileStreamRotator.getStream(this.streamOptions);
    }
    this.stream.write(data);
  }
}

// LogFileStream not available in browser, will use default browser console output instead.
const stream = isBrowser || isTest ? undefined : new LogFileStream();

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
