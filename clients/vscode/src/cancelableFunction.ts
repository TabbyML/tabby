import { CancellationToken, CancellationTokenSource } from "vscode";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type CancelableFunction = (...args: any[]) => Promise<unknown>;

/**
 * Create a function that wrap a cancelable function fn, the returned function will cancel the previous call (if any) and then call fn.
 * Take care of the returned promise may be rejected by auto cancellation, according to the fn implementation.
 * @param fn the function to wrap
 * @param getToken a function that extract the token from the arguments, can be omitted if the token is the first argument
 * @param setToken a function that set the token in the arguments, can be omitted if the token is the first argument
 */
export function wrapCancelableFunction<Fn extends CancelableFunction>(
  fn: Fn,
  getToken: (args: Parameters<Fn>) => CancellationToken | undefined = (args) => args[0],
  setToken: (args: Parameters<Fn>, token: CancellationToken) => Parameters<Fn> = (args, token) =>
    [token, ...args.slice(1)] as Parameters<Fn>,
): Fn {
  let currentCancelSource: CancellationTokenSource | null = null;
  return function (...args: Parameters<Fn>): ReturnType<Fn> {
    currentCancelSource?.cancel();
    const cancelSource = new CancellationTokenSource();
    const token = getToken(args);
    if (token?.isCancellationRequested) {
      cancelSource.cancel();
    }
    token?.onCancellationRequested(() => {
      cancelSource.cancel();
    });
    currentCancelSource = cancelSource;
    const updatedArgs = setToken(args, cancelSource.token);
    return fn(...updatedArgs) as ReturnType<Fn>;
  } as unknown as Fn;
}
