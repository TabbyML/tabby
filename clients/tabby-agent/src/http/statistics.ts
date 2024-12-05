export type RequestStatsResult = {
  values: number[];
  stats: { total: number; timeouts: number; responses: number; averageResponseTime: number };
};

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
