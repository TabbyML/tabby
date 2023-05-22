import { Agent } from "./Agent";
import { CancelablePromise } from "./generated";

const agent = new Agent();

type FunctionName = "cancelRequest" | "setServerUrl" | "getServerUrl" | "getCompletions" | "postEvent";

interface AgentRequest {
  0: number; // Agent request id
  1: {
    func: FunctionName;
    args: any;
  };
}

interface AgentResponse {
  0: number; // Matched request id, set to 0 if no request matched
  1: any; // Response data
}

const pendingRequests: { [id: number]: CancelablePromise<any> } = {};

process.stdin.on("data", async (data) => {
  let req: AgentRequest | null = null;
  try {
    req = JSON.parse(data.toString());
  } catch (e) {
    process.stderr.write(JSON.stringify(e, Object.getOwnPropertyNames(e)) + "\n");
    return;
  }

  const resp: AgentResponse = [req[0], {}];
  try {
    switch (req[1].func) {
      case "cancelRequest":
        const pendingRequest = pendingRequests[req[1].args.id];
        if (!!pendingRequest) {
          pendingRequest.cancel();
        }
        resp[1] = true;
        break;
      case "setServerUrl":
        resp[1] = agent.setServerUrl(req[1].args.url);
        break;
      case "getServerUrl":
        resp[1] = agent.getServerUrl();
        break;
      case "getCompletions":
        pendingRequests[req[0]] = agent.getCompletions(req[1].args);
        resp[1] = await pendingRequests[req[0]];
        delete pendingRequests[req[0]];
        break;
      case "postEvent":
        pendingRequests[req[0]] = agent.postEvent(req[1].args);
        resp[1] = await pendingRequests[req[0]];
        delete pendingRequests[req[0]];
        break;
      default:
        throw new Error(`Unknown function: ${req[1].func}`);
    }
  } catch (e) {
    process.stderr.write(JSON.stringify(e, Object.getOwnPropertyNames(e)) + "\n");
  } finally {
    process.stdout.write(JSON.stringify(resp) + "\n");
  }
});

agent.on("statusChanged", (status) => {
  const resp: AgentResponse = [0, { event: "statusChanged", status }];
  process.stdout.write(JSON.stringify(resp) + "\n");
});
