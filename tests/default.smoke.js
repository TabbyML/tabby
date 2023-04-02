import http from "k6/http";
import { check } from "k6";

export default function () {
  const payload = JSON.stringify({
    prompt: "def binarySearch(arr, left, right, x):\n    mid = (left +",
  });
  const headers = { "Content-Type": "application/json" };
  const res = http.post("http://localhost:5000/v1/completions", payload, {
    headers,
  });
  check(res, { success: (r) => r.status === 200 });
}
