import type { Logger } from "./type";
import type { Connection } from "vscode-languageserver";
import type { Configurations } from "../config";
import type { ConfigData } from "../config/type";
import EventEmitter from "events";
import { getFileLogger, PinoLogger } from "./fileLogger";

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
    if (destination instanceof PinoLogger) {
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

export class LoggerManager {
  private static instance: LoggerManager;
  public static getInstance(): LoggerManager {
    if (!LoggerManager.instance) {
      LoggerManager.instance = new LoggerManager();
    }
    return LoggerManager.instance;
  }

  private logDestinations = new Destinations();
  private fileLogger: PinoLogger | undefined;

  getLogger(tag: string): Logger {
    return new MultiDestinationLogger(this.logDestinations, tag);
  }

  preInitialize(config: Configurations): void {
    this.fileLogger = getFileLogger();
    if (this.fileLogger) {
      this.logDestinations.attach(this.fileLogger);
      this.setFileLoggerLevel(config.getMergedConfig().logs.level);
    }
    config.on("updated", (configData: ConfigData) => {
      if (configData.logs) {
        this.setFileLoggerLevel(configData.logs.level);
      }
    });
  }

  private setFileLoggerLevel(level: string) {
    if (this.fileLogger) {
      this.fileLogger.level = level;
    }
  }

  attachLogger(logger: Logger) {
    this.logDestinations.attach(logger);
  }

  attachLspConnection(connection: Connection) {
    this.attachLogger({
      error: (msg: string, error: any) => {
        const errorMsg =
          error instanceof Error
            ? `[${error.name}] ${error.message} \n${error.stack}`
            : JSON.stringify(error, undefined, 2);
        connection.console.error(`${msg} ${errorMsg}`);
      },
      warn: (msg: string) => {
        connection.console.warn(msg);
      },
      info: (msg: string) => {
        connection.console.info(msg);
      },
      debug: (msg: string) => {
        connection.console.debug(msg);
      },
      trace: () => {},
    });
  }
}

export function getLogger(tag: string): Logger {
  return LoggerManager.getInstance().getLogger(tag);
}
