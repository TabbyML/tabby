import { Univariate } from "stats-logscale";

export type CompletionProviderStatsEntry = {
  triggerMode: "auto" | "manual";
};

export type CompletionRequestStatsEntry = {
  latency: number; // ms, NaN if timeout/canceled
  canceled: boolean;
  timeout: boolean;
};

export class CompletionStats {
  private autoCompletionCount = 0;
  private manualCompletionCount = 0;

  private eventMap = new Map<string, number>();

  private completionRequestLatencyStats = new Univariate();
  private completionRequestCanceledCount = 0;
  private completionRequestTimeoutCount = 0;

  addProviderStatsEntry(value: CompletionProviderStatsEntry): void {
    const { triggerMode } = value;
    if (triggerMode === "auto") {
      this.autoCompletionCount += 1;
    } else {
      this.manualCompletionCount += 1;
    }
  }

  addRequestStatsEntry(value: CompletionRequestStatsEntry): void {
    const { canceled, timeout, latency } = value;
    if (canceled) {
      this.completionRequestCanceledCount += 1;
    } else if (timeout) {
      this.completionRequestTimeoutCount += 1;
    } else {
      this.completionRequestLatencyStats.add(latency);
    }
  }

  addEvent(event: string): void {
    const count = this.eventMap.get(event) || 0;
    this.eventMap.set(event, count + 1);
  }

  reset(): void {
    this.autoCompletionCount = 0;
    this.manualCompletionCount = 0;

    this.eventMap = new Map<string, number>();

    this.completionRequestLatencyStats = new Univariate();
    this.completionRequestCanceledCount = 0;
    this.completionRequestTimeoutCount = 0;
  }

  // stats for anonymous usage report
  stats(): Record<string, any> {
    const eventCount = Object.fromEntries(
      Array.from(this.eventMap.entries()).map(([key, value]) => ["count_" + key, value]),
    );
    return {
      completion: {
        count_auto: this.autoCompletionCount,
        count_manual: this.manualCompletionCount,
        ...eventCount,
      },
      completion_request: {
        count: this.completionRequestLatencyStats.count(),
        latency_avg: this.completionRequestLatencyStats.mean(),
        latency_p50: this.completionRequestLatencyStats.percentile(50),
        latency_p95: this.completionRequestLatencyStats.percentile(95),
        latency_p99: this.completionRequestLatencyStats.percentile(99),
      },
      completion_request_canceled: {
        count: this.completionRequestCanceledCount,
      },
      completion_request_timeout: {
        count: this.completionRequestTimeoutCount,
      },
    };
  }
}

class LastN {
  private readonly values: number[] = [];

  constructor(private readonly maxSize: number) {}

  add(value: number): void {
    this.values.push(value);
    if (this.values.length > this.maxSize) {
      this.values.shift();
    }
  }

  getValues(): number[] {
    return this.values;
  }
}

type RequestStatsResult = {
  values: number[];
  stats: { total: number; timeouts: number; responses: number; averageResponseTime: number };
};

export class RequestStats {
  static readonly config = {
    windowSize: 10,
    checks: {
      disable: false,
      // Mark status as healthy if the latency is less than the threshold for each latest windowSize requests.
      healthy: { windowSize: 1, latency: 3000 },
      // If there is at least {count} requests, and the average response time is higher than the {latency}, show warning
      slowResponseTime: { latency: 5000, count: 1 },
      // If there is at least {count} timeouts, and the timeout rate is higher than the {rate}, show warning
      highTimeoutRate: { rate: 0.5, count: 1 },
    },
  };

  static check(result: RequestStatsResult): "healthy" | "highTimeoutRate" | "slowResponseTime" | null {
    if (this.config.checks.disable) {
      return null;
    }
    const config = this.config.checks;

    const {
      values: latencies,
      stats: { total, timeouts, responses, averageResponseTime },
    } = result;

    if (
      latencies
        .slice(-Math.min(this.config.windowSize, config.healthy.windowSize))
        .every((latency) => latency < config.healthy.latency)
    ) {
      return "healthy";
    }
    if (timeouts / total > config.highTimeoutRate.rate && timeouts >= config.highTimeoutRate.count) {
      return "highTimeoutRate";
    }
    if (averageResponseTime > config.slowResponseTime.latency && responses >= config.slowResponseTime.count) {
      return "slowResponseTime";
    }
    return null;
  }

  private latencies: LastN = new LastN(RequestStats.config.windowSize);

  add(entry: number): void {
    this.latencies.add(entry);
  }

  reset(): void {
    this.latencies = new LastN(RequestStats.config.windowSize);
  }

  // stats for "highTimeoutRate" | "slowResponseTime" warning
  stats(): RequestStatsResult {
    const latencies = this.latencies.getValues();
    const timeouts = latencies.filter((latency) => Number.isNaN(latency));
    const responses = latencies.filter((latency) => !Number.isNaN(latency));
    const averageResponseTime = responses.reduce((acc, latency) => acc + latency, 0) / responses.length;
    return {
      values: latencies,
      stats: {
        total: latencies.length,
        timeouts: timeouts.length,
        responses: responses.length,
        averageResponseTime,
      },
    };
  }
}
