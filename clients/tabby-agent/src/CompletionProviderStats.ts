import { Univariate } from "stats-logscale";
import { rootLogger } from "./logger";

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
  private readonly logger = rootLogger.child({ component: "CompletionProviderStats" });

  private autoCompletionCount = 0;
  private manualCompletionCount = 0;
  private cacheHitCount = 0;
  private cacheMissCount = 0;

  private completionRequestLatencyStats = new Univariate();
  private completionRequestCanceledStats = new Average();
  private completionRequestTimeoutCount = 0;

  private recentCompletionRequestLatencies = new Windowed(10);

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

  reset() {
    this.autoCompletionCount = 0;
    this.manualCompletionCount = 0;
    this.cacheHitCount = 0;
    this.cacheMissCount = 0;
    this.completionRequestLatencyStats = new Univariate();
    this.completionRequestCanceledStats = new Average();
    this.completionRequestTimeoutCount = 0;
  }

  resetWindowed() {
    this.recentCompletionRequestLatencies = new Windowed(10);
  }

  // stats for anonymous usage report
  stats() {
    return {
      completion: {
        count_auto: this.autoCompletionCount,
        count_manual: this.manualCompletionCount,
        cache_hit: this.cacheHitCount,
        cache_miss: this.cacheMissCount,
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

  static check(windowed: WindowedStats): "healthy" | "highTimeoutRate" | "slowResponseTime" | null {
    const {
      values: latencies,
      stats: { total, timeouts, responses, averageResponseTime },
    } = windowed;
    // if the recent 3 requests all have latency less than 3s
    if (latencies.slice(-3).every((latency) => latency < 3000)) {
      return "healthy";
    }
    // if the recent requests timeout percentage is more than 50%, at least 3 timeouts
    if (timeouts / total > 0.5 && timeouts >= 3) {
      return "highTimeoutRate";
    }
    // if average response time is more than 4s, at least 3 requests
    if (responses >= 3 && averageResponseTime > 4000) {
      return "slowResponseTime";
    }
    return null;
  }
}
