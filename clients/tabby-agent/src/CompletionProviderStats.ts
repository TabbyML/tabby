import { Univariate } from "stats-logscale";

export type CompletionProviderStatsEntry = {
  triggerMode: "auto" | "manual";
  cacheHit: boolean;
  aborted: boolean;
  requestSent: boolean;
  requestLatency: number; // ms, NaN if timeout
  requestCanceled: boolean;
  requestTimeout: boolean;
};

class Average {
  private sum = 0;
  private quantity = 0;

  add(value: number): void {
    this.sum += value;
    this.quantity += 1;
  }

  mean(): number | undefined {
    if (this.quantity === 0) {
      return undefined;
    }
    return this.sum / this.quantity;
  }

  count(): number {
    return this.quantity;
  }
}

class Windowed {
  private readonly maxSize: number;
  private readonly values: number[] = [];

  constructor(maxSize: number) {
    this.maxSize = maxSize;
  }

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

type WindowedStats = {
  values: number[];
  stats: { total: number; timeouts: number; responses: number; averageResponseTime: number };
};

export class CompletionProviderStats {
  private config = {
    windowSize: 10,
    checks: {
      disable: false,
      // Mark status as healthy if the latency is less than the threshold for each latest windowSize requests.
      healthy: { windowSize: 3, latency: 2400 },
      // If there is at least {count} requests, and the average response time is higher than the {latency}, show warning
      slowResponseTime: { latency: 3200, count: 3 },
      // If there is at least {count} timeouts, and the timeout rate is higher than the {rate}, show warning
      highTimeoutRate: { rate: 0.5, count: 3 },
    },
  };

  private autoCompletionCount = 0;
  private manualCompletionCount = 0;
  private cacheHitCount = 0;
  private cacheMissCount = 0;

  private eventMap = new Map<string, number>();

  private completionRequestLatencyStats = new Univariate();
  private completionRequestCanceledStats = new Average();
  private completionRequestTimeoutCount = 0;

  private recentCompletionRequestLatencies: Windowed = new Windowed(this.config.windowSize);

  updateConfigByRequestTimeout(timeout: number) {
    this.config.checks.healthy.latency = timeout * 0.6;
    this.config.checks.slowResponseTime.latency = timeout * 0.8;
    this.resetWindowed();
  }

  add(value: CompletionProviderStatsEntry): void {
    const { triggerMode, cacheHit, aborted, requestSent, requestLatency, requestCanceled, requestTimeout } = value;
    if (!aborted) {
      if (triggerMode === "auto") {
        this.autoCompletionCount += 1;
      } else {
        this.manualCompletionCount += 1;
      }
      if (cacheHit) {
        this.cacheHitCount += 1;
      } else {
        this.cacheMissCount += 1;
      }
    }
    if (requestSent) {
      if (requestCanceled) {
        this.completionRequestCanceledStats.add(requestLatency);
      } else if (requestTimeout) {
        this.completionRequestTimeoutCount += 1;
      } else {
        this.completionRequestLatencyStats.add(requestLatency);
      }
      if (!requestCanceled) {
        this.recentCompletionRequestLatencies.add(requestLatency);
      }
    }
  }

  addEvent(event: string): void {
    const count = this.eventMap.get(event) || 0;
    this.eventMap.set(event, count + 1);
  }

  reset() {
    this.autoCompletionCount = 0;
    this.manualCompletionCount = 0;
    this.cacheHitCount = 0;
    this.cacheMissCount = 0;

    this.eventMap = new Map<string, number>();

    this.completionRequestLatencyStats = new Univariate();
    this.completionRequestCanceledStats = new Average();
    this.completionRequestTimeoutCount = 0;
  }

  resetWindowed() {
    this.recentCompletionRequestLatencies = new Windowed(this.config.windowSize);
  }

  // stats for anonymous usage report
  stats() {
    const eventCount = Object.fromEntries(
      Array.from(this.eventMap.entries()).map(([key, value]) => ["count_" + key, value]),
    );
    return {
      completion: {
        count_auto: this.autoCompletionCount,
        count_manual: this.manualCompletionCount,
        cache_hit: this.cacheHitCount,
        cache_miss: this.cacheMissCount,
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
        count: this.completionRequestCanceledStats.count(),
        latency_avg: this.completionRequestCanceledStats.mean(),
      },
      completion_request_timeout: {
        count: this.completionRequestTimeoutCount,
      },
    };
  }

  // stats for "highTimeoutRate" | "slowResponseTime" warning
  windowed(): WindowedStats {
    const latencies = this.recentCompletionRequestLatencies.getValues();
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

  check(windowed: WindowedStats): "healthy" | "highTimeoutRate" | "slowResponseTime" | null {
    if (this.config.checks.disable) {
      return null;
    }
    const config = this.config.checks;

    const {
      values: latencies,
      stats: { total, timeouts, responses, averageResponseTime },
    } = windowed;

    if (
      latencies
        .slice(-Math.max(this.config.windowSize, config.healthy.windowSize))
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
}
