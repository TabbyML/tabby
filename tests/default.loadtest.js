import http from "k6/http";
import { check, group, sleep, abortTest } from "k6";

export const options = {
  stages: [
    { duration: "5s", target: 8 },
    { duration: "20s", target: 8 },
    { duration: "5s", target: 0 },
  ],
  // Below thresholds are tested against TabbyML/StarCoder-1B served by NVIDIA T4 GPU.
  thresholds: {
    http_req_failed: ['rate<0.01'], // http errors should be less than 1%
    http_req_duration: ["med<1800", "avg<1800", "p(95)<2000"],
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
