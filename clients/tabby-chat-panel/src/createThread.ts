/* eslint-disable style/yield-star-spacing */
/* eslint-disable unicorn/error-message */
/* eslint-disable ts/consistent-type-imports */
/* eslint-disable style/operator-linebreak */
/* eslint-disable ts/method-signature-style */
/* eslint-disable sort-imports */

/* eslint-disable style/brace-style */
/* eslint-disable antfu/if-newline */
/* eslint-disable style/comma-dangle */
/* eslint-disable style/semi */
/* eslint-disable style/member-delimiter-style */
/* eslint-disable style/quotes */

/* eslint-disable unused-imports/no-unused-vars */

import {
  AnyFunction,
  createBasicEncoder,
  isMemoryManageable,
  RELEASE_METHOD,
  RETAIN_METHOD,
  RETAINED_BY,
  StackFrame,
  Thread,
  ThreadEncoder,
  ThreadEncoderApi,
  ThreadTarget,
} from "@quilted/threads";

const CALL = 0;
const RESULT = 1;
const TERMINATE = 2;
const RELEASE = 3;
const FUNCTION_APPLY = 5;
const FUNCTION_RESULT = 6;
export const CHECK_MESSAGE = "quilt.threads.ping";
export const RESPONSE_MESSAGE = "quilt.threads.pong";

interface MessageMap {
  [CALL]: [string, string | number, any];
  [RESULT]: [string, Error?, any?];
  [TERMINATE]: [];
  [RELEASE]: [string];
  [FUNCTION_APPLY]: [string, string, any];
  [FUNCTION_RESULT]: [string, Error?, any?];
}

type MessageData = {
  [K in keyof MessageMap]: [K, MessageMap[K]];
}[keyof MessageMap];

export interface ThreadOptions<
  Self = Record<string, never>,
  Target = Record<string, never>,
> {
  expose?: Self;
  signal?: AbortSignal;
  encoder?: ThreadEncoder;
  callable?: (keyof Target)[];
  uuid?(): string;
  onMethodUnavailable?: (method: string) => void;
}

export function createThread<
  Self = Record<string, never>,
  Target = Record<string, never>,
>(
  target: ThreadTarget,
  {
    expose,
    callable,
    signal,
    uuid = defaultUuid,
    encoder = createBasicEncoder(),
    onMethodUnavailable,
  }: ThreadOptions<Self, Target> = {}
): Thread<Target> {
  let terminated = false;
  const activeApi = new Map<string | number, AnyFunction>();
  const functionsToId = new Map<AnyFunction, string>();
  const idsToFunction = new Map<string, AnyFunction>();
  const idsToProxy = new Map<string, AnyFunction>();

  if (expose) {
    for (const key of Object.keys(expose)) {
      const value = expose[key as keyof typeof expose];
      if (typeof value === "function") activeApi.set(key, value);
    }
  }

  const callIdsToResolver = new Map<
    string,
    (
      ...args: MessageMap[typeof FUNCTION_RESULT] | MessageMap[typeof RESULT]
    ) => void
  >();

  const call = createCallable<Thread<Target>>(handlerForCall, callable);

  const encoderApi: ThreadEncoderApi = {
    functions: {
      add(func) {
        let id = functionsToId.get(func);
        if (id == null) {
          id = uuid();
          functionsToId.set(func, id);
          idsToFunction.set(id, func);
        }
        return id;
      },
      get(id) {
        let proxy = idsToProxy.get(id);
        if (proxy) return proxy;

        let retainCount = 0;
        let released = false;

        const release = () => {
          retainCount -= 1;
          if (retainCount === 0) {
            released = true;
            idsToProxy.delete(id);
            send(RELEASE, [id]);
          }
        };

        const retain = () => {
          retainCount += 1;
        };

        proxy = (...args: any[]) => {
          if (released) {
            return Promise.resolve(undefined);
          }

          if (!idsToProxy.has(id)) {
            return Promise.resolve(undefined);
          }

          const [encoded, transferable] = encoder.encode(args, encoderApi);
          const callId = uuid();
          const done = waitForResult(callId);
          send(FUNCTION_APPLY, [callId, id, encoded], transferable);
          return done;
        };

        Object.defineProperties(proxy, {
          [RELEASE_METHOD]: { value: release, writable: false },
          [RETAIN_METHOD]: { value: retain, writable: false },
          [RETAINED_BY]: { value: new Set(), writable: false },
        });

        idsToProxy.set(id, proxy);
        return proxy;
      },
    },
  };

  function send<Type extends keyof MessageMap>(
    type: Type,
    args: MessageMap[Type],
    transferables?: Transferable[]
  ) {
    if (terminated) return;
    target.send([type, args], transferables);
  }

  async function listener(rawData: unknown) {
    const isThreadMessageData =
      Array.isArray(rawData) &&
      typeof rawData[0] === "number" &&
      (rawData[1] == null || Array.isArray(rawData[1]));

    if (!isThreadMessageData) return;

    const data = rawData as MessageData;

    switch (data[0]) {
      case CALL: {
        const stackFrame = new StackFrame();
        const [id, property, args] = data[1];
        const func = activeApi.get(property);

        try {
          if (func == null) {
            throw new Error(
              `No '${property}' method is exposed on this endpoint`
            );
          }

          const result = await func(
            ...(encoder.decode(args, encoderApi, [stackFrame]) as any[])
          );
          const [encoded, transferables] = encoder.encode(result, encoderApi);
          send(RESULT, [id, undefined, encoded], transferables);
        } catch (error) {
          const { name, message, stack } = error as Error;
          send(RESULT, [id, { name, message, stack }]);
        } finally {
          stackFrame.release();
        }
        break;
      }
      case RESULT: {
        resolveCall(...data[1]);
        break;
      }
      case TERMINATE: {
        terminate();
        break;
      }
      case RELEASE: {
        const [id] = data[1];
        const func = idsToFunction.get(id);
        if (func) {
          idsToFunction.delete(id);
          functionsToId.delete(func);
        }
        break;
      }
      case FUNCTION_APPLY: {
        const [callId, funcId, args] = data[1];
        const stackFrame = new StackFrame();

        try {
          const func = idsToFunction.get(funcId);
          if (func == null) {
            const [encoded] = encoder.encode(undefined, encoderApi);
            send(FUNCTION_RESULT, [callId, undefined, encoded]);
          } else {
            const result = await func(
              ...(encoder.decode(
                args,
                encoderApi,
                isMemoryManageable(func)
                  ? [...func[RETAINED_BY], stackFrame]
                  : [stackFrame]
              ) as any[])
            );
            const [encoded, transferables] = encoder.encode(result, encoderApi);
            send(FUNCTION_RESULT, [callId, undefined, encoded], transferables);
          }
        } finally {
          stackFrame.release();
        }
        break;
      }
      case FUNCTION_RESULT: {
        resolveCall(...data[1]);
        break;
      }
    }
  }

  function handlerForCall(property: string | number | symbol) {
    return (...args: any[]) => {
      if (
        terminated ||
        (typeof property !== "string" && typeof property !== "number")
      ) {
        return Promise.resolve(undefined);
      }

      if (property === "hasCapability") {
        const methodToTest = args[0];
        const testId = uuid();
        const testPromise = waitForResult(testId);
        send(CALL, [testId, methodToTest, encoder.encode([], encoderApi)[0]]);
        return testPromise.then(
          () => {
            return true;
          },
          (error) => {
            if (
              error.message?.includes(
                `No '${methodToTest}' method is exposed on this endpoint`
              )
            ) {
              return false;
            }
            throw error;
          }
        );
      }
      const id = uuid();
      const done = waitForResult(id);
      const [encoded, transferables] = encoder.encode(args, encoderApi);
      send(CALL, [id, property, encoded], transferables);
      return done;
    };
  }

  function waitForResult(id: string) {
    const promise = new Promise<any>((resolve, reject) => {
      callIdsToResolver.set(id, (_, errorResult, value) => {
        if (errorResult == null) {
          resolve(encoder.decode(value, encoderApi));
        } else {
          const error = new Error();
          Object.assign(error, errorResult);
          reject(error);
        }
      });
    });

    Object.defineProperty(promise, Symbol.asyncIterator, {
      async *value() {
        const result = await promise;
        Object.defineProperty(result, Symbol.asyncIterator, {
          value: () => result,
        });
        yield* result;
      },
    });
    return promise;
  }

  function resolveCall(...args: MessageMap[typeof RESULT]) {
    const callId = args[0];
    const resolver = callIdsToResolver.get(callId);
    if (resolver) {
      resolver(...args);
      callIdsToResolver.delete(callId);
    }
  }

  function terminate() {
    if (terminated) return;
    for (const id of callIdsToResolver.keys()) {
      resolveCall(id, undefined, encoder.encode(undefined, encoderApi)[0]);
    }
    terminated = true;
    activeApi.clear();
    callIdsToResolver.clear();
    functionsToId.clear();
    idsToFunction.clear();
    idsToProxy.clear();
  }

  signal?.addEventListener(
    "abort",
    () => {
      send(TERMINATE, []);
      terminate();
    },
    { once: true }
  );

  target.listen(listener, { signal });

  return call;
}

function defaultUuid() {
  return `${uuidSegment()}-${uuidSegment()}-${uuidSegment()}-${uuidSegment()}`;
}

function uuidSegment() {
  return Math.floor(Math.random() * Number.MAX_SAFE_INTEGER).toString(16);
}

function createCallable<T>(
  handlerForCall: (property: string | number | symbol) => AnyFunction,
  callable?: (keyof T)[]
): T {
  let call: any;

  if (callable == null) {
    if (typeof Proxy !== "function") {
      return {} as T;
    }

    const cache = new Map<string | number | symbol, AnyFunction>();
    call = new Proxy(
      {},
      {
        get(_target, property) {
          if (cache.has(property)) {
            return cache.get(property);
          }
          const handler = handlerForCall(property);
          cache.set(property, handler);
          return handler;
        },
      }
    );
  } else {
    call = {};
    for (const method of callable) {
      Object.defineProperty(call, method, {
        value: handlerForCall(method),
        writable: false,
        configurable: true,
        enumerable: true,
      });
    }
  }

  return call;
}
