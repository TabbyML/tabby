import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
    stages: [
        {duration: '3s', target: 5},
    ],
    hosts: {
        'api.tabbyml.com': __ENV.TABBY_API_HOST || "localhost:5000"
    },
};
const SLEEP_DURATION = 1;

export default function () {
  const payload = JSON.stringify({
    prompt: "def binarySearch(arr, left, right, x):\n    mid = (left +",
  });
  const headers = { "Content-Type": "application/json" };
  const res = http.post("http://api.tabbyml.com/v1/completions", payload, {
    headers,
  });
  check(res, { success: (r) => r.status === 200 });
  sleep(SLEEP_DURATION)
}
