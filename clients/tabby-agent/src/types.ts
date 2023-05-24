import { CancelablePromise, ChoiceEvent, CompletionEvent, CompletionRequest, CompletionResponse } from "./generated";

export interface AgentFunction {
  setServerUrl(url: string): string;
  getServerUrl(): string;
  getStatus(): "connecting" | "ready" | "disconnected";
  getCompletions(request: CompletionRequest): CancelablePromise<CompletionResponse>;
  postEvent(event: ChoiceEvent | CompletionEvent): CancelablePromise<any>;
}

export type StatusChangedEvent = {
  event: "statusChanged";
  status: "connecting" | "ready" | "disconnected";
}

export type AgentEvent = StatusChangedEvent;
export const agentEventNames: AgentEvent['event'][] = ["statusChanged"];

export interface AgentEventEmitter {
  on<T extends AgentEvent>(eventName: T["event"], callback: (event: T) => void): this;
}

export type Agent = AgentFunction & AgentEventEmitter;
