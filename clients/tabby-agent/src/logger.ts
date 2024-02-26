import path from "path";
import os from "os";
import pino from "pino";
import * as FileStreamRotator from "file-stream-rotator";
import { isBrowser, isTest, testLogDebug } from "./env";

export interface LogFn {
  (msg: string, ...args: any[]): void;
}

export interface Logger {
  error: LogFn;
  warn: LogFn;
  info: LogFn;
  debug: LogFn;
  trace: LogFn;
}

export type ObjLogFn = {
  <T extends object>(obj: T, msg?: string, ...args: any[]): void;
  (obj: unknown, msg?: string, ...args: any[]): void;
  (msg: string, ...args: any[]): void;
};

export interface ObjLogger {
  error: ObjLogFn;
  warn: ObjLogFn;
  info: ObjLogFn;
  debug: ObjLogFn;
  trace: ObjLogFn;
}

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
const logFileStream = isBrowser || isTest ? undefined : new LogFileStream();
const pinoOptions = { serializers: { error: pino.stdSerializers.err } };
const rootBasicLogger = logFileStream ? pino(pinoOptions, logFileStream) : pino(pinoOptions);
if (isTest && testLogDebug) {
  rootBasicLogger.level = "debug";
} else {
  rootBasicLogger.level = "silent";
}
export const allBasicLoggers = [rootBasicLogger];
rootBasicLogger.onChild = (child: pino.Logger) => {
  allBasicLoggers.push(child);
};

function toObjLogFn(logFn: LogFn): ObjLogFn {
  return (...args: any[]) => {
    const arg0 = args.shift();
    if (typeof arg0 === "string") {
      logFn(arg0, ...args);
    } else {
      const arg1 = args.shift();
      if (typeof arg1 === "string") {
        logFn(arg1, ...args, arg0);
      } else {
        logFn(arg0, arg1, ...args);
      }
    }
  };
}

function withComponent(logFn: LogFn, component: string): LogFn {
  return (msg: string, ...args: any[]) => {
    logFn(`[${component}] ${msg ?? ""}`, ...args);
  };
}

export const extraLogger = {
  loggers: [] as Logger[],
  child(component: string): ObjLogger {
    const buildLogFn = (level: keyof Logger) => {
      const logFn = (...args: any[]) => {
        const arg0 = args.shift();
        this.loggers.forEach((logger) => logger[level](arg0, ...args));
      };
      return toObjLogFn(withComponent(logFn, component));
    };
    return {
      error: buildLogFn("error"),
      warn: buildLogFn("warn"),
      info: buildLogFn("info"),
      debug: buildLogFn("debug"),
      trace: buildLogFn("trace"),
    };
  },
};

export function logger(component: string): ObjLogger {
  const basic = rootBasicLogger.child({ component });
  const extra = extraLogger.child(component);
  const all = [basic, extra];
  const buildLogFn = (level: keyof ObjLogger) => {
    return (...args: any[]) => {
      const arg0 = args.shift();
      all.forEach((logger) => logger[level](arg0, ...args));
    };
  };
  return {
    error: buildLogFn("error"),
    warn: buildLogFn("warn"),
    info: buildLogFn("info"),
    debug: buildLogFn("debug"),
    trace: buildLogFn("trace"),
  };
}
