import type { components as TabbyApiComponents } from "tabby-openapi/compatible";

export type LatencyStatistics = {
  values: number[];
  metrics: { total: number; timeouts: number; responses: number; averageResponseTime: number };
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

export class LatencyTracker {
  private windowSize: number;
  private latencies: LastN;

  constructor(options?: { windowSize: number }) {
    this.windowSize = options?.windowSize ?? 10;
    this.latencies = new LastN(this.windowSize);
  }

  // add a latency entry, add NaN for timeouts
  add(entry: number): void {
    this.latencies.add(entry);
  }

  reset(): void {
    this.latencies = new LastN(this.windowSize);
  }

  calculateLatencyStatistics(): LatencyStatistics {
    const latencies = this.latencies.getValues();
    const timeouts = latencies.filter((latency) => Number.isNaN(latency));
    const responses = latencies.filter((latency) => !Number.isNaN(latency));
    const averageResponseTime = responses.reduce((acc, latency) => acc + latency, 0) / responses.length;
    return {
      values: latencies,
      metrics: {
        total: latencies.length,
        timeouts: timeouts.length,
        responses: responses.length,
        averageResponseTime,
      },
    };
  }
}

export function analyzeMetrics(
  latencyStatistics: LatencyStatistics,
): "healthy" | "highTimeoutRate" | "slowResponseTime" | null {
  const rules = {
    // Mark status as healthy if the latency is less than the threshold for each latest windowSize requests.
    healthy: { windowSize: 1, latency: 3000 },
    // If there is at least {count} requests, and the average response time is higher than the {latency}, show warning
    slowResponseTime: { latency: 5000, count: 1 },
    // If there is at least {count} timeouts, and the timeout rate is higher than the {rate}, show warning
    highTimeoutRate: { rate: 0.5, count: 1 },
  };

  const {
    values: latencies,
    metrics: { total, timeouts, responses, averageResponseTime },
  } = latencyStatistics;

  if (
    latencies
      .slice(-Math.min(latencies.length, rules.healthy.windowSize))
      .every((latency) => latency < rules.healthy.latency)
  ) {
    return "healthy";
  }
  if (timeouts / total > rules.highTimeoutRate.rate && timeouts >= rules.highTimeoutRate.count) {
    return "highTimeoutRate";
  }
  if (averageResponseTime > rules.slowResponseTime.latency && responses >= rules.slowResponseTime.count) {
    return "slowResponseTime";
  }
  return null;
}

export function buildHelpMessageForLatencyIssue(
  issue: "highTimeoutRate" | "slowResponseTime",
  data?:
    | {
        latencyStatistics: LatencyStatistics;
        endpoint?: string;
        serverHealth?: TabbyApiComponents["schemas"]["HealthState"];
      }
    | undefined,
  format?: "plaintext" | "markdown" | "html",
): string | undefined {
  const outputFormat = format ?? "plaintext";
  const metrics = data?.latencyStatistics.metrics;
  const serverHealth = data?.serverHealth;
  const endpoint = data?.endpoint;

  let statsMessage = "";
  if (issue == "slowResponseTime") {
    if (metrics && metrics["responses"] && metrics["averageResponseTime"]) {
      statsMessage = `The average response time of recent ${metrics["responses"]} completion requests is ${Number(
        metrics["averageResponseTime"],
      ).toFixed(0)}ms.<br/><br/>`;
    }
  }

  if (issue == "highTimeoutRate") {
    if (metrics && metrics["total"] && metrics["timeouts"]) {
      statsMessage = `${metrics["timeouts"]} of ${metrics["total"]} completion requests timed out.<br/><br/>`;
    }
  }

  let helpMessageForRunningLargeModelOnCPU = "";
  if (serverHealth?.device === "cpu" && serverHealth?.model?.match(/[0-9.]+B$/)) {
    helpMessageForRunningLargeModelOnCPU +=
      `Your Tabby server is running model <i>${serverHealth?.model}</i> on CPU. ` +
      "This model may be performing poorly due to its large parameter size, please consider trying smaller models or switch to GPU. " +
      "You can find a list of recommend models in the <a href='https://tabby.tabbyml.com/docs/'>online documentation</a>.<br/>";
  }
  let commonHelpMessage = "";
  if (helpMessageForRunningLargeModelOnCPU.length == 0) {
    commonHelpMessage += `<li>The running model <i>${
      serverHealth?.model ?? ""
    }</i> may be performing poorly due to its large parameter size. `;
    commonHelpMessage +=
      "Please consider trying smaller models. You can find a list of recommend models in the <a href='https://tabby.tabbyml.com/docs/'>online documentation</a>.</li>";
  }
  if (endpoint) {
    const host = new URL(endpoint).host;
    if (!(host.startsWith("localhost") || host.startsWith("127.0.0.1") || host.startsWith("0.0.0.0"))) {
      commonHelpMessage += "<li>A poor network connection. Please check your network and proxy settings.</li>";
      commonHelpMessage += "<li>Server overload. Please contact your Tabby server administrator for assistance.</li>";
    }
  }
  let helpMessage = "";
  if (helpMessageForRunningLargeModelOnCPU.length > 0) {
    helpMessage += helpMessageForRunningLargeModelOnCPU + "<br/>";
    if (commonHelpMessage.length > 0) {
      helpMessage += "Other possible causes of this issue: <br/><ul>" + commonHelpMessage + "</ul>";
    }
  } else {
    // commonHelpMessage should not be empty here
    helpMessage += "Possible causes of this issue: <br/><ul>" + commonHelpMessage + "</ul>";
  }

  const message = statsMessage + helpMessage;
  if (outputFormat == "html") {
    return message;
  }
  if (outputFormat == "markdown") {
    return message
      .replace(/<br\/>/g, " \n")
      .replace(/<i>(.*?)<\/i>/g, "*$1*")
      .replace(/<a\s+(?:[^>]*?\s+)?href=["']([^"']+)["'][^>]*>([^<]+)<\/a>/g, "[$2]($1)")
      .replace(/<ul[^>]*>(.*?)<\/ul>/g, "$1")
      .replace(/<li[^>]*>(.*?)<\/li>/g, "- $1 \n");
  } else {
    return message
      .replace(/<br\/>/g, " \n")
      .replace(/<i>(.*?)<\/i>/g, "$1")
      .replace(/<a[^>]*>(.*?)<\/a>/g, "$1")
      .replace(/<ul[^>]*>(.*?)<\/ul>/g, "$1")
      .replace(/<li[^>]*>(.*?)<\/li>/g, "- $1 \n");
  }
}
