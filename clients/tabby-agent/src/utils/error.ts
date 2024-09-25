// Http Error
export class HttpError extends Error {
  public readonly status: number;
  public readonly statusText: string;
  public readonly response: Response;

  constructor(response: Response) {
    super(`${response.status} ${response.statusText}`);
    this.name = "HttpError";
    this.status = response.status;
    this.statusText = response.statusText;
    this.response = response;
  }
}

export class MutexAbortError extends Error {
  constructor() {
    super("Aborted due to new request.");
    this.name = "AbortError";
  }
}

export function isTimeoutError(error: any) {
  return (
    (error instanceof Error && error.name === "TimeoutError") ||
    (error instanceof HttpError && [408, 499].includes(error.status))
  );
}

export function isCanceledError(error: any) {
  return error instanceof Error && error.name === "AbortError";
}

export function isUnauthorizedError(error: any) {
  return error instanceof HttpError && [401, 403].includes(error.status);
}

export function errorToString(error: Error) {
  let message = error.message || error.toString();
  if (error.cause instanceof Error) {
    message += "\nCaused by: " + errorToString(error.cause);
  }
  return message;
}
