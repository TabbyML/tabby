import { TabbyClient } from "./TabbyClient";

const tabby = TabbyClient.getInstance();

interface VimRequest {
  0: number; // Vim request id
  1: {
    func: string;
    args?: any;
  };
}

interface VimResponse {
  0: number; // Matched request id, set to 0 if no request matched
  1: any; // Response data
}

process.stdin.on("data", async (data) => {
  try {
    const req: VimRequest = JSON.parse(data.toString());
    const resp: VimResponse = [req[0], {}];
    if (req[1].func in tabby && typeof tabby[req[1].func] === "function") {
      const func = tabby[req[1].func] as Function;
      const args = Array.isArray(req[1].args) ? req[1].args : [req[1].args];
      resp[1] = await func.call(tabby, ...args);
    }
    process.stdout.write(JSON.stringify(resp) + "\n");
  } catch (e) {
    // FIXME: log errors
  }
});

tabby.on("statusChanged", (status) => {
  const resp: VimResponse = [0, { event: "statusChanged", status }];
  process.stdout.write(JSON.stringify(resp) + "\n");
});
