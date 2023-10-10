import http from "k6/http";
import { check, group, sleep } from "k6";

export const options = {
  stages: [
    { duration: "5s", target: 10 }, // simulate ramp-up of traffic from 1 to 10 users over 30s.
    { duration: "30s", target: 10 }, // stay at 10 users for 10 minutes
    { duration: "5s", target: 0 }, // ramp-down to 0 users
  ],
  hosts: {
    "api.tabbyml.com": __ENV.TABBY_API_HOST || "localhost:5000",
  },
  thresholds: {
    http_req_duration: ["p(99)<1000"], // 99% of requests must complete below 1000ms
  },
};

export default () => {
  const payload = JSON.stringify({
    prompt: "def binarySearch(arr, left, right, x):\n    mid = (left +",
  });
  const headers = { "Content-Type": "application/json" };
  const res = http.post("https://tabbyml-tabby-template-space.hf.space/v1/completions", payload, {
    headers,
  });
  check(res, { success: (r) => r.status === 200 });
  sleep(0.5);
};
