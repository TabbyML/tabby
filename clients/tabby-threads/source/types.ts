import type {
  RELEASE_METHOD,
  RETAIN_METHOD,
  ENCODE_METHOD,
  RETAINED_BY,
} from "./constants.ts";

/**
 * A thread represents a target JavaScript environment that exposes a set
 * of callable, asynchronous methods. The thread takes care of automatically
 * encoding and decoding its arguments and return values, so you can interact
 * with it as if its methods were implemented in the same environment as your
 * own code.
 */
export type Thread<Target> = {
  [K in keyof Target]: Target[K] extends (...args: any[]) => infer ReturnType
    ? ReturnType extends Promise<any> | AsyncGenerator<any, any, any>
      ? Target[K]
      : never
    : never;
} & {
  
  /**
   * A method that used to get all exposed methods of the opposite thread.
   */
  _requestMethods(): Promise<string[]>;
};

/**
 * An object backing a `Thread` that provides the message-passing interface
 * that allows communication to flow between environments. This message-passing
 * interface is based on the [`postMessage` interface](https://developer.mozilla.org/en-US/docs/Web/API/Window/postMessage),
 * which is easily adaptable to many JavaScript objects and environments.
 */
export interface ThreadTarget {
  /**
   * Sends a message to the target thread. The message will be encoded before sending,
   * and the consumer may also pass an array of "transferable" objects that should be
   * transferred (rather than copied) to the other environment, if supported.
   */
  send(message: any, transferables?: Transferable[]): void;

  /**
   * Listens for messages coming in to the thread. This method must call the provided
   * listener for each message as it is received. The thread will then decode the message
   * and handle its content. This method may be passed an `AbortSignal` to abort the
   * listening process.
   */
  listen(listener: (value: any) => void, options: {signal?: AbortSignal}): void;
}

/**
 * A mapped object type that takes an object with methods, and converts it into the
 * an object with the same methods that can be called over a thread.
 */

/**
 * An object that can retain a reference to a `MemoryManageable` object.
 */
export interface MemoryRetainer {
  add(manageable: MemoryManageable): void;
}

/**
 * An object transferred between threads that must have its memory manually managed,
 * in order to release the reference to a corresponding object on the original thread.
 */
export interface MemoryManageable {
  readonly [RETAINED_BY]: Set<MemoryRetainer>;
  [RETAIN_METHOD](): void;
  [RELEASE_METHOD](): void;
}

/**
 * An object that can encode and decode values communicated between two threads.
 */
export interface ThreadEncoder {
  /**
   * Encodes a value before sending it to another thread. Should return a tuple where
   * the first item is the encoded value, and the second item is an array of elements
   * that can be transferred to the other thread, instead of being copied.
   */
  encode(value: unknown, api: ThreadEncoderApi): [any, Transferable[]?];

  /**
   * Decodes a value received from another thread.
   */
  decode(
    value: unknown,
    api: ThreadEncoderApi,
    retainedBy?: Iterable<MemoryRetainer>,
  ): unknown;
}

export interface ThreadEncoderApi {
  /**
   * Controls how the thread encoder will handle functions.
   */
  functions?: {
    /**
     * Retrieve a function by its serialized ID. This function will be called while
     * decoding responses from the other "side" of a thread. The implementer of this
     * API should return a proxy function that will call the function on the other
     * thread, or `undefined` to prevent the function from being being decoded.
     */
    get(id: string): AnyFunction | undefined;

    /**
     * Stores a function during encoding. The implementer of this API should return
     * a unique ID for the function, or `undefined` to prevent the function from
     * being encoded.
     */
    add(func: AnyFunction): string | undefined;
  };
}

/**
 * An object that provides a custom process to encode its value.
 */
export interface ThreadEncodable {
  [ENCODE_METHOD](api: {encode(value: any): unknown}): any;
}

export type AnyFunction = Function;
