import type {
  Thread,
  ThreadTarget,
  ThreadEncoder,
  ThreadEncoderApi,
  AnyFunction,
} from "../types.ts";

import {
  RELEASE_METHOD,
  RETAINED_BY,
  RETAIN_METHOD,
  StackFrame,
  isMemoryManageable,
} from "../memory";
import { createBasicEncoder } from "../encoding/basic";

export type { ThreadTarget };

/**
 * Options to customize the creation of a `Thread` instance.
 */
export interface ThreadOptions<
  Self = Record<string, never>,
  Target = Record<string, never>,
> {
  /**
   * Methods to expose on this thread, so that they are callable on the paired thread.
   * This should be an object, with each member of that object being a function. Remember
   * that these functions will become asynchronous when called over the thread boundary.
   */
  expose?: Self;

  /**
   * An `AbortSignal` that controls whether the thread is active or not. When aborted,
   * the thread will no longer send any messages to the underlying object, will stop
   * listening for messages from that object, and will clean up any memory associated
   * with in-progress communication between the threads.
   */
  signal?: AbortSignal;

  /**
   * An object that will encode and decode messages sent between threads. If not
   * provided, a default implementation created with `createBasicEncoder()` will be used instead.
   */
  encoder?: ThreadEncoder;

  /**
   * A list of callable methods exposed on the paired `Thread`. This option is
   * required if you want to call methods and your environment does not support
   * the `Proxy` constructor. When the `Proxy` constructor is available, `createThread()`
   * will forward all method calls to the paired `Thread` by default.
   */
  callable?: (keyof Target)[];

  /**
   * A function for generating unique identifiers. Unique identifiers are used by
   * some encoding and decoding operations to maintain stable references to objects
   * transferred between the threads. If not provided, a simple default implementation
   * will be used instead.
   */
  uuid?(): string;
}

const CALL = 0;
const RESULT = 1;
const TERMINATE = 2;
const RELEASE = 3;
const FUNCTION_APPLY = 5;
const FUNCTION_RESULT = 6;
const CHECK_CAPABILITY = 7;
const EXPOSE_LIST = 8;

interface MessageMap {
  [CALL]: [string, string | number, any];
  [RESULT]: [string, Error?, any?];
  [TERMINATE]: [];
  [RELEASE]: [string];
  [FUNCTION_APPLY]: [string, string, any];
  [FUNCTION_RESULT]: [string, Error?, any?];
  [CHECK_CAPABILITY]: [string, string];
  [EXPOSE_LIST]: [string, string[]]; // Request to exchange methods: [callId, our_methods]
  // The other side will respond with their methods via RESULT
}

type MessageData = {
  [K in keyof MessageMap]: [K, MessageMap[K]];
}[keyof MessageMap];

/**
 * Creates a thread from any object that conforms to the `ThreadTarget`
 * interface.
 */
export async function createThread<
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
  }: ThreadOptions<Self, Target> = {}
): Promise<Thread<Target>> {
  console.log("[createThread] Initializing with options:", {
    hasExpose: !!expose,
    hasCallable: !!callable,
    hasSignal: !!signal,
    exposeMethods: expose ? Object.keys(expose) : [],
  });

  let terminated = false;
  const activeApi = new Map<string | number, AnyFunction>();
  const functionsToId = new Map<AnyFunction, string>();
  const idsToFunction = new Map<string, AnyFunction>();
  const idsToProxy = new Map<string, AnyFunction>();

  // Cache for the other side's methods
  let theirMethodsCache: string[] | null = null;

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

  // Create functions for method exchange
  const exchangeMethods = () => {
    console.log("[createThread] Starting expose list exchange");
    const ourMethods = Array.from(activeApi.keys()).map(String);
    console.log("[createThread] Our expose list:", ourMethods);

    const id = uuid();
    console.log("[createThread] Setting up expose list resolver");

    // This will be called when we receive the RESULT with other side's methods
    callIdsToResolver.set(id, (_, __, value) => {
      const theirMethods = encoder.decode(value, encoderApi) as string[];
      console.log(
        "[createThread] Got RESULT with other side's methods:",
        theirMethods
      );
      // Cache their methods
      theirMethodsCache = theirMethods;
      console.log("[createThread] Expose list exchange completed");
    });

    // Send EXPOSE_LIST with our methods
    console.log("[createThread] Sending EXPOSE_LIST with our methods");
    send(EXPOSE_LIST, [id, ourMethods]);
  };

  // Create a function to request methods from the other side
  const requestMethods = async () => {
    // If we have cached methods and connection is still active, return them
    if (theirMethodsCache !== null && !terminated) {
      console.log(
        "[createThread] Returning cached methods:",
        theirMethodsCache
      );
      return theirMethodsCache;
    }

    console.log("[createThread] Requesting methods from other side");
    const id = uuid();

    // Create a promise that will resolve with the other side's methods
    const methodsPromise = new Promise<string[]>((resolve) => {
      callIdsToResolver.set(id, (_, __, value) => {
        const theirMethods = encoder.decode(value, encoderApi) as string[];
        console.log(
          "[createThread] Got RESULT with other side's methods:",
          theirMethods
        );
        // Cache the methods for future use
        theirMethodsCache = theirMethods;
        resolve(theirMethods);
      });
    });

    // Send EXPOSE_LIST with empty methods array to request other side's methods
    console.log("[createThread] Sending method request");
    send(EXPOSE_LIST, [id, []]);

    return methodsPromise;
  };

  // Create proxy for method calls
  console.log("[createThread] Creating proxy without waiting for response");
  const call = createCallable<Thread<Target>>(handlerForCall, callable, {
    exchangeMethods,
    requestMethods,
  });

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
            throw new Error(
              "You attempted to call a function that was already released."
            );
          }

          if (!idsToProxy.has(id)) {
            throw new Error(
              "You attempted to call a function that was already revoked."
            );
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

  const terminate = () => {
    if (terminated) return;

    for (const id of callIdsToResolver.keys()) {
      resolveCall(id, new ThreadTerminatedError());
    }

    terminated = true;
    activeApi.clear();
    callIdsToResolver.clear();
    functionsToId.clear();
    idsToFunction.clear();
    idsToProxy.clear();
    theirMethodsCache = null; // Clear the cache when connection is terminated
  };

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

  function send<Type extends keyof MessageMap>(
    type: Type,
    args: MessageMap[Type],
    transferables?: Transferable[]
  ) {
    if (terminated) {
      console.log("[createThread] Not sending message - thread terminated");
      return;
    }
    console.log("[createThread] Sending message:", {
      type,
      args,
      transferables,
    });
    target.send([type, args], transferables);
  }

  async function listener(rawData: unknown) {
    console.log("[createThread] Received raw data:", rawData);

    // FIXME: don't ignore anything, just for testing now
    const isThreadMessageData =
      Array.isArray(rawData) &&
      typeof rawData[0] === "number" &&
      (rawData[1] == null || Array.isArray(rawData[1]));

    if (!isThreadMessageData) {
      console.log("[createThread] Invalid message format, ignoring:,", rawData);
      return;
    }

    const data = rawData as MessageData;
    console.log("[createThread] Processing message type:", data[0]);

    switch (data[0]) {
      case TERMINATE: {
        terminate();
        break;
      }
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
        const [id, error, value] = data[1];
        console.log("[createThread] Received RESULT message:", {
          id,
          error,
          value,
        });

        // If this is a response to our EXPOSE_LIST
        const resolver = callIdsToResolver.get(id);
        if (resolver) {
          console.log("[createThread] Found resolver for RESULT");
          if (error) {
            console.log("[createThread] Error in RESULT:", error);
          } else {
            const methods = encoder.decode(value, encoderApi);
            console.log("[createThread] Decoded methods from RESULT:", methods);
          }
        }

        resolveCall(...data[1]);
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
      case FUNCTION_RESULT: {
        resolveCall(...data[1]);
        break;
      }
      case FUNCTION_APPLY: {
        const [callId, funcId, args] = data[1];

        const stackFrame = new StackFrame();

        try {
          const func = idsToFunction.get(funcId);

          if (func == null) {
            throw new Error(
              "You attempted to call a function that was already released."
            );
          }

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
        } catch (error) {
          const { name, message, stack } = error as Error;
          send(FUNCTION_RESULT, [callId, { name, message, stack }]);
        } finally {
          stackFrame.release();
        }

        break;
      }
      case CHECK_CAPABILITY: {
        const [id, methodToCheck] = data[1];
        const hasMethod = activeApi.has(methodToCheck);
        send(RESULT, [id, undefined, encoder.encode(hasMethod, encoderApi)[0]]);
        break;
      }
      case EXPOSE_LIST: {
        const [id, theirMethods] = data[1];
        console.log(
          "[createThread] Received EXPOSE_LIST with their methods:",
          theirMethods
        );

        // Store their methods for future use
        const theirMethodsList = theirMethods as string[];
        console.log("[createThread] Stored their methods:", theirMethodsList);

        // Send back our methods as RESULT
        const ourMethods = Array.from(activeApi.keys()).map(String);
        console.log(
          "[createThread] Sending RESULT with our methods:",
          ourMethods
        );

        send(RESULT, [
          id,
          undefined,
          encoder.encode(ourMethods, encoderApi)[0],
        ]);

        console.log(
          "[createThread] Expose list exchange completed for this side"
        );
        break;
      }
    }
  }

  function handlerForCall(property: string | number | symbol) {
    return (...args: any[]) => {
      try {
        if (terminated) {
          throw new ThreadTerminatedError();
        }

        if (typeof property !== "string" && typeof property !== "number") {
          throw new Error(
            `Can't call a symbol method on a thread: ${property.toString()}`
          );
        }

        // hasCapability is a special method that checks if a method is exposed on the other thread
        if (property === "hasCapability") {
          const methodToCheck = args[0];
          const id = uuid();
          const done = waitForResult(id);
          send(CHECK_CAPABILITY, [id, methodToCheck]);
          return done;
        }

        //normal call
        const id = uuid();
        const done = waitForResult(id);
        const [encoded, transferables] = encoder.encode(args, encoderApi);

        send(CALL, [id, property, encoded], transferables);

        return done;
      } catch (error) {
        return Promise.reject(error);
      }
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
}

class ThreadTerminatedError extends Error {
  constructor() {
    super("You attempted to call a function on a terminated thread.");
  }
}

function defaultUuid() {
  return `${uuidSegment()}-${uuidSegment()}-${uuidSegment()}-${uuidSegment()}`;
}

function uuidSegment() {
  return Math.floor(Math.random() * Number.MAX_SAFE_INTEGER).toString(16);
}

function createCallable<T>(
  handlerForCall: (
    property: string | number | symbol
  ) => AnyFunction | undefined,
  callable?: (keyof T)[],
  methods?: {
    exchangeMethods: () => void;
    requestMethods: () => Promise<string[]>;
  }
): T {
  console.log("[createCallable] Creating callable with methods:", callable);
  let call: any;

  if (callable == null) {
    if (typeof Proxy !== "function") {
      throw new Error(
        `You must pass an array of callable methods in environments without Proxies.`
      );
    }

    const cache = new Map<string | number | symbol, AnyFunction | undefined>();

    call = new Proxy(
      {},
      {
        get(_target, property) {
          if (property === "then") return undefined;
          if (property === "exchangeMethods") {
            console.log("[createCallable] Accessing exchangeMethods");
            return methods?.exchangeMethods;
          }
          if (property === "requestMethods") {
            console.log("[createCallable] Accessing requestMethods");
            return methods?.requestMethods;
          }
          console.log("[createCallable] Accessing property:", property);
          if (cache.has(property)) {
            console.log("[createCallable] Using cached handler for:", property);
            return cache.get(property);
          }

          console.log("[createCallable] Creating new handler for:", property);
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
