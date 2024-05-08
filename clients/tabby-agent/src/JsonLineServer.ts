import readline from "readline";
import { AgentFunction, AgentEvent, Agent, agentEventNames } from "./Agent";
import { getLogger } from "./logger";
import { isCanceledError } from "./utils";

type AgentFunctionRequest<T extends keyof AgentFunction> = [
  id: number,
  data: {
    func: T;
    args: Parameters<AgentFunction[T]>;
  },
];

type CancellationRequest = [
  id: number,
  data: {
    func: "cancelRequest";
    args: [id: number];
  },
];

type JsonLineRequest = AgentFunctionRequest<keyof AgentFunction> | CancellationRequest;

type AgentFunctionResponse<T extends keyof AgentFunction> = [
  id: number, // Matched request id
  data: ReturnType<AgentFunction[T]> | null,
];

type AgentEventNotification = [
  id: 0, // Always 0
  data: AgentEvent,
];

type CancellationResponse = [
  id: number, // Matched request id
  data: boolean | null,
];

type JsonLineResponse = AgentFunctionResponse<keyof AgentFunction> | AgentEventNotification | CancellationResponse;

/**
 * Every request and response should be single line JSON string and end with a newline.
 */
export class JsonLineServer {
  private readonly process: NodeJS.Process = process;
  private readonly inStream: NodeJS.ReadStream = process.stdin;
  private readonly outStream: NodeJS.WriteStream = process.stdout;
  private readonly logger = getLogger("JsonLineServer");

  private abortControllers: { [id: string]: AbortController } = {};

  private agent?: Agent;

  constructor() {}

  private async handleLine(line: string) {
    let request: JsonLineRequest;
    try {
      request = JSON.parse(line) as JsonLineRequest;
    } catch (error) {
      this.logger.error(`Failed to parse request: ${line}`, error);
      return;
    }
    this.logger.debug(`Processing request ${request[0]} ${request[1].func}.`);
    this.logger.trace("Request data:", { request });
    const response = await this.handleRequest(request);
    this.sendResponse(response);
    this.logger.debug(`Completed processing request ${request[0]}.`);
    this.logger.trace("Response data:", { response });
  }

  private async handleRequest(request: JsonLineRequest): Promise<JsonLineResponse> {
    let requestId: number = 0;
    const response: JsonLineResponse = [0, null];
    const abortController = new AbortController();
    try {
      if (!this.agent) {
        throw new Error(`Agent not bound.\n`);
      }
      requestId = request[0];
      response[0] = requestId;

      const funcName = request[1].func;
      if (funcName === "cancelRequest") {
        response[1] = this.cancelRequest(request as CancellationRequest);
      } else {
        const func = this.agent[funcName];
        if (!func) {
          throw new Error(`Unknown function: ${funcName}`);
        }
        const args = request[1].args;
        // If the last argument is an object and has `signal` property, replace it with the abort signal.
        if (args.length > 0 && typeof args[args.length - 1] === "object" && args[args.length - 1]["signal"]) {
          this.abortControllers[requestId] = abortController;
          args[args.length - 1]["signal"] = abortController.signal;
        }
        // @ts-expect-error TS2684: FIXME
        response[1] = await func.apply(this.agent, args);
      }
    } catch (error) {
      if (isCanceledError(error)) {
        this.logger.debug(`Request ${request[0]} canceled.`);
      } else {
        this.logger.error(`Failed to handle request ${request[0]}`, error);
      }
    } finally {
      if (this.abortControllers[requestId]) {
        delete this.abortControllers[requestId];
      }
    }
    return response;
  }

  private cancelRequest(request: CancellationRequest): boolean {
    const targetId = request[1].args[0];
    const controller = this.abortControllers[targetId];
    if (controller) {
      controller.abort();
      return true;
    }
    return false;
  }

  private sendResponse(response: JsonLineResponse): void {
    this.outStream.write(JSON.stringify(response) + "\n");
  }

  bind(agent: Agent): void {
    this.agent = agent;
    for (const eventName of agentEventNames) {
      this.agent.on(eventName, (event) => {
        this.sendResponse([0, event]);
      });
    }
  }

  listen() {
    readline.createInterface({ input: this.inStream }).on("line", (line) => {
      this.handleLine(line);
    });

    ["SIGTERM", "SIGINT"].forEach((sig) => {
      this.process.on(sig, async () => {
        if (this.agent && this.agent.getStatus() !== "finalized") {
          await this.agent.finalize();
        }
        this.process.exit(0);
      });
    });
  }
}
