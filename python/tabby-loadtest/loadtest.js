import http from "k6/http";
import { check, sleep } from "k6";
import { textSummary } from "https://jslib.k6.io/k6-utils/1.4.0/index.js";

const PARALLELISM = parseInt(__ENV.PARALLELISM);

export const options = {
  stages: [
    { duration: "1s", target: PARALLELISM },
    { duration: "30s", target: PARALLELISM },
  ],
  // Below thresholds are tested against TabbyML/StarCoder-1B served by NVIDIA T4 GPU.
  thresholds: {
    http_req_failed: ['rate<0.001'],
    http_req_duration: ["med<1800", "avg<1800", "p(90)<2500", "p(95)<3000"],
  },
};

export default () => {
  const payload = JSON.stringify({
    language: "python",
    segments: {
      prefix: "def binarySearch(arr, left, right, x):\n    mid = (left +"
    },
  });
  const headers = { "Content-Type": "application/json" };
  const res = http.post(`${__ENV.TABBY_API_HOST}/v1/completions`, payload, {
    headers,
  });
  check(res, { success: (r) => r.status === 200 });
  sleep(0.5);
};

export function handleSummary(data) {
  const avg_latency = data.metrics.http_req_duration.values.avg / 1000;
  const med_latency = data.metrics.http_req_duration.values.med / 1000;
  const p90_latency = data.metrics.http_req_duration.values["p(90)"] / 1000;
  const p95_latency = data.metrics.http_req_duration.values["p(95)"] / 1000;
  const qps = PARALLELISM / avg_latency;

  return {
    "metrics.txt": `${rounded(qps)},${rounded(avg_latency)},${rounded(med_latency)},${rounded(p90_latency)},${rounded(p95_latency)}`
  };
}

function rounded(x) {
  return Math.round(x * 100) / 100;
}