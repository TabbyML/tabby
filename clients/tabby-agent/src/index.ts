import { TabbyClient } from "./TabbyClient";
import { getFunction } from "./utils";
import { CancelablePromise, CancelError, ApiError } from "./generated";
const tabby = TabbyClient.getInstance();

interface VimRequest {
  0: number; // Vim request id
  1: {
    func: string;
    args?: any;
    cancelPendingRequest?: boolean; // cancel pending request that called the same API
  };
}

interface VimResponse {
  0: number; // Matched request id, set to 0 if no request matched
  1: any; // Response data
}

const pendingPromise : { [key: string]: CancelablePromise<any> } = {};

process.stdin.on("data", async (data) => {
  try {
    const req: VimRequest = JSON.parse(data.toString());
    const resp: VimResponse = [req[0], {}];
    const func = getFunction(tabby, req[1].func);
    if (func) {
      const args = Array.isArray(req[1].args) ? req[1].args : [req[1].args];
      const result = func(...args);
      if (result instanceof CancelablePromise && req[1].func.startsWith("api.")) {
        // Async API calls
        if (req[1].cancelPendingRequest && pendingPromise[req[1].func]) {
          pendingPromise[req[1].func].cancel();
        }
        pendingPromise[req[1].func] = result;
        resp[1] = await result
          .then((response: any) => {
            tabby.changeStatus("ready");
            return response;
          })
          .catch((e: CancelError) => {
            return null;
          })
          .catch((e: ApiError) => {
            process.stderr.write(JSON.stringify(e, Object.getOwnPropertyNames(e)) + "\n");
            tabby.changeStatus("disconnected");
            return null;
          });
        pendingPromise[req[1].func] = null;
      } else if (result instanceof Promise) {
        // Async calls (non-API)
        resp[1] = await result;
      } else {
        // Sync calls
        resp[1] = result;
      }
    }
    process.stdout.write(JSON.stringify(resp) + "\n");
  } catch (e) {
    process.stderr.write(JSON.stringify(e, Object.getOwnPropertyNames(e)) + "\n");
  }
});

tabby.on("statusChanged", (status) => {
  const resp: VimResponse = [0, { event: "statusChanged", status }];
  process.stdout.write(JSON.stringify(resp) + "\n");
});
