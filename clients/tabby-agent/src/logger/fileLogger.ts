import type { Logger } from "./type";
import path from "path";
import os from "os";
import * as FileStreamRotator from "file-stream-rotator";
import pino from "pino";
import { isBrowser, isTest } from "../env";

export class LogFileStream implements pino.DestinationStream {
  private stream?: ReturnType<typeof FileStreamRotator.getStream>;

  write(data: string): void {
    if (!this.stream) {
      // Rotating file locate at `~/.tabby-client/agent/logs/`.
      const logDir = path.join(os.homedir(), ".tabby-client", "agent", "logs");
      const now = new Date();
      const dateString = `${now.getFullYear()}${(now.getMonth() + 1).toString().padStart(2, "0")}${now.getDate().toString().padStart(2, "0")}`;
      const timeString = `${now.getHours().toString().padStart(2, "0")}${now.getMinutes().toString().padStart(2, "0")}${now.getSeconds().toString().padStart(2, "0")}`;
      const logFilePathName = path.join(logDir, dateString, `${timeString}-${process.pid.toString()}`);
      this.stream = FileStreamRotator.getStream({
        filename: logFilePathName,
        size: "10M",
        max_logs: "30d",
        end_stream: true,
        audit_file: path.join(logDir, "audit.json"),
        extension: ".log",
      });
    }
    this.stream.write(data);
  }
}

export class PinoLogger implements Logger {
  private childLoggers: pino.Logger[] = [];

  constructor(private baseLogger: pino.Logger) {
    this.baseLogger.onChild = (child: pino.Logger) => {
      this.childLoggers.push(child);
    };
  }

  get level(): string {
    return this.baseLogger.level;
  }

  set level(level: string) {
    this.baseLogger.level = level;
    this.childLoggers.forEach((child) => {
      child.level = level;
    });
  }

  child(tag: string): Logger {
    return new PinoLogger(this.baseLogger.child({ tag }));
  }

  error(msg: string, error: any) {
    this.baseLogger.error({ error }, msg);
  }
  warn(msg: string) {
    this.baseLogger.warn(msg);
  }
  info(msg: string) {
    this.baseLogger.info(msg);
  }
  debug(msg: string) {
    this.baseLogger.debug(msg);
  }
  trace(msg: string, verbose?: any) {
    this.baseLogger.trace(verbose, msg);
  }
}

export function getFileLogger(): PinoLogger | undefined {
  const fileLogger =
    isBrowser || isTest
      ? undefined
      : new PinoLogger(pino({ serializers: { error: pino.stdSerializers.err } }, new LogFileStream()));

  return fileLogger;
}
