import path from "path";
import os from "os";
import EventEmitter from "events";
import pino from "pino";
import * as FileStreamRotator from "file-stream-rotator";
import { isBrowser, isTest } from "./env";

export interface Logger {
  error: (msg: string, error: any) => void;
  warn: (msg: string) => void;
  info: (msg: string) => void;
  debug: (msg: string) => void;
  trace: (msg: string, verbose?: any) => void;
}

class LogFileStream implements pino.DestinationStream {
  private stream?: pino.DestinationStream;

  write(data: string): void {
    if (!this.stream) {
      this.stream = FileStreamRotator.getStream({
        // Rotating file locate at `~/.tabby-client/agent/logs/`.
        filename: path.join(os.homedir(), ".tabby-client", "agent", "logs", "%DATE%"),
        frequency: "daily",
        size: "10M",
        max_logs: "30d",
        extension: ".log",
        create_symlink: true,
      });
    }
    this.stream.write(data);
  }
}

class TaggedLogger implements Logger {
  constructor(
    private baseLogger: Logger,
    private tag: string,
  ) {}

  private tagMsg(msg: string): string {
    return `[${this.tag}] ${msg}`;
  }

  error(msg: string, error: any): void {
    this.baseLogger.error(this.tagMsg(msg), { error });
  }
  warn(msg: string): void {
    this.baseLogger.warn(this.tagMsg(msg));
  }
  info(msg: string): void {
    this.baseLogger.info(this.tagMsg(msg));
  }
  debug(msg: string): void {
    this.baseLogger.debug(this.tagMsg(msg));
  }
  trace(msg: string, verbose?: any): void {
    this.baseLogger.trace(this.tagMsg(msg), verbose);
  }
}

class JsonLineLogger implements Logger {
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
    return new JsonLineLogger(this.baseLogger.child({ tag }));
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

class Destinations extends EventEmitter {
  constructor(public readonly destinations: Logger[] = []) {
    super();
  }

  attach(...destinations: Logger[]) {
    this.destinations.push(...destinations);
    this.emit("newAttached", destinations);
  }
}

class MultiDestinationLogger implements Logger {
  private loggers: Logger[] = [];

  constructor(
    public destinations: Destinations,
    private readonly tag?: string,
  ) {
    this.loggers.push(...destinations.destinations.map((destination) => this.createTaggedLogger(destination)));
    destinations.on("newAttached", (destinations: Logger[]) => {
      this.loggers.push(...destinations.map((destination) => this.createTaggedLogger(destination)));
    });
  }

  private createTaggedLogger(destination: Logger): Logger {
    if (!this.tag) {
      return destination;
    }
    if (destination instanceof JsonLineLogger) {
      return destination.child(this.tag);
    }
    return new TaggedLogger(destination, this.tag);
  }

  error(msg: string, error: any) {
    this.loggers.forEach((logger) => logger.error(msg, error));
  }
  warn(msg: string) {
    this.loggers.forEach((logger) => logger.warn(msg));
  }
  info(msg: string) {
    this.loggers.forEach((logger) => logger.info(msg));
  }
  debug(msg: string) {
    this.loggers.forEach((logger) => logger.debug(msg));
  }
  trace(msg: string, verbose?: any) {
    this.loggers.forEach((logger) => logger.trace(msg, verbose));
  }
}

export const logDestinations = new Destinations();

export const fileLogger =
  isBrowser || isTest
    ? undefined
    : new JsonLineLogger(pino({ serializers: { error: pino.stdSerializers.err } }, new LogFileStream()));

if (fileLogger) {
  logDestinations.attach(fileLogger);
}

export function getLogger(tag: string): Logger {
  return new MultiDestinationLogger(logDestinations, tag);
}
