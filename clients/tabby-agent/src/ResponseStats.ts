import { EventEmitter } from "events";
import { rootLogger } from "./logger";

export type ResponseStatsEntry = {
  name: string;
  status: number;
  responseTime: number;
  error?: any;
};

export type ResponseStatsStrategy = {
  maxSize: number;
  stats: Record<string, (entries: ResponseStatsEntry[]) => number>;
  checks: {
    name: string;
    check: (entries: ResponseStatsEntry[], stats: Record<string, number>) => boolean;
  }[];
};

export const completionResponseTimeStatsStrategy = {
  maxSize: 50,
  stats: {
    total: (entries: ResponseStatsEntry[]) => entries.length,
    responses: (entries: ResponseStatsEntry[]) => entries.filter((entry) => entry.status === 200).length,
    timeouts: (entries: ResponseStatsEntry[]) => entries.filter((entry) => entry.error?.isTimeoutError).length,
    averageResponseTime: (entries: ResponseStatsEntry[]) =>
      entries.filter((entry) => entry.status === 200).reduce((acc, entry) => acc + entry.responseTime, 0) /
      entries.length,
  },
  checks: [
    // check in order and emit the first event that matches
    // if all the last 5 entries have status 200 and response time less than 3s
    {
      name: "healthy",
      check: (entries: ResponseStatsEntry[], stats) => {
        const recentEntries = entries.slice(-5);
        return recentEntries.every((entry) => entry.status === 200 && entry.responseTime < 3000);
      },
    },
    // if TimeoutError percentage is more than 50%, at least 3 requests
    {
      name: "highTimeoutRate",
      check: (entries: ResponseStatsEntry[], stats) => {
        if (stats.total < 3) {
          return false;
        }
        return stats.timeouts / stats.total > 0.5;
      },
    },
    // if average response time is more than 4s, at least 5 requests
    {
      name: "slowResponseTime",
      check: (entries: ResponseStatsEntry[], stats) => {
        if (stats.responses < 5) {
          return false;
        }
        return stats.averageResponseTime > 4000;
      },
    },
  ],
};

export class ResponseStats extends EventEmitter {
  private readonly logger = rootLogger.child({ component: "ResponseStats" });
  private strategy: ResponseStatsStrategy = {
    maxSize: 0,
    stats: {},
    checks: [],
  };

  private entries: ResponseStatsEntry[] = [];

  constructor(strategy: ResponseStatsStrategy) {
    super();
    this.strategy = strategy;
  }

  push(entry: ResponseStatsEntry): void {
    this.entries.push(entry);
    if (this.entries.length > this.strategy.maxSize) {
      this.entries.shift();
    }
    const stats = this.stats();
    for (const check of this.strategy.checks) {
      if (check.check(this.entries, stats)) {
        this.logger.debug({ check: check.name, stats }, "Check condition met");
        this.emit(check.name, stats);
      }
    }
  }

  stats(): Record<string, number> {
    const result: Record<string, number> = {};
    for (const [name, stats] of Object.entries(this.strategy.stats)) {
      result[name] = stats(this.entries);
    }
    return result;
  }

  check(): string | null {
    const stats = this.stats();
    for (const check of this.strategy.checks) {
      if (check.check(this.entries, stats)) {
        return check.name;
      }
    }
    return null;
  }
}
