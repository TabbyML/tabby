import { ENCODE_METHOD, TRANSFERABLE } from "../constants";
import type {
  ThreadEncoder,
  ThreadEncoderApi,
  ThreadEncodable,
} from "../types";
import {
  isBasicObject,
  isMemoryManageable,
  type MemoryRetainer,
} from "../memory";

const FUNCTION = "_@f";
const MAP = "_@m";
const SET = "_@s";
const URL_ID = "_@u";
const DATE = "_@d";
const REGEXP = "_@r";
const ASYNC_ITERATOR = "_@i";

export interface ThreadEncoderOptions {
  /**
   * Customizes the encoding of each value in a passed message.
   */
  encode?(
    value: unknown,
    defaultEncode: (value: unknown) => [any, Transferable[]?]
  ): [any, Transferable[]?];

  /**
   * Customizes the decoding of each value in a passed message.
   */
  decode?(
    value: unknown,
    defaultDecode: (
      value: unknown,
      retainedBy?: Iterable<MemoryRetainer>
    ) => unknown,
    retainedBy?: Iterable<MemoryRetainer>
  ): unknown;
}

/**
 * Creates an encoder that converts most common JavaScript types into a format
 * that can be transferred via message passing.
 */
export function createBasicEncoder({
  encode: encodeOverride,
  decode: decodeOverride,
}: ThreadEncoderOptions = {}): ThreadEncoder {
  return {
    encode,
    decode,
  };

  type EncodeResult = ReturnType<ThreadEncoder["encode"]>;

  interface EncodeContext {
    api: ThreadEncoderApi;
    seen: Map<any, EncodeResult>;
    encode: Parameters<NonNullable<ThreadEncoderOptions["encode"]>>[1];
  }

  function encode(value: unknown, api: ThreadEncoderApi): EncodeResult {
    const context: EncodeContext = {
      api,
      seen: new Map(),
      encode: (value: any) => encodeInternal(value, context, true),
    };

    return encodeInternal(value, context);
  }

  function encodeInternal(
    value: unknown,
    context: EncodeContext,
    isFromOverride = false
  ): EncodeResult {
    const { seen, api, encode } = context;

    if (!isFromOverride && encodeOverride) {
      return encodeOverride(value, encode);
    }

    if (value == null) return [value];

    const seenValue = seen.get(value);
    if (seenValue) return seenValue;

    seen.set(value, [undefined]);

    if (typeof value === "object") {
      if ((value as any)[TRANSFERABLE]) {
        const result: EncodeResult = [value, [value as any]];
        seen.set(value, result);
        return result;
      }

      const transferables: Transferable[] = [];
      const encodeValue = (value: any) => {
        const [fieldValue, nestedTransferables = []] = encodeInternal(
          value,
          context
        );
        transferables.push(...nestedTransferables);
        return fieldValue;
      };

      if (typeof (value as any)[ENCODE_METHOD] === "function") {
        const result = (value as ThreadEncodable)[ENCODE_METHOD]({
          encode: encodeValue,
        });

        const fullResult: EncodeResult = [result, transferables];
        seen.set(value, fullResult);

        return fullResult;
      }

      if (Array.isArray(value)) {
        const result = value.map((item) => encodeValue(item));
        const fullResult: EncodeResult = [result, transferables];
        seen.set(value, fullResult);
        return fullResult;
      }

      // TODO: avoid this if using a `structuredClone` postMessage-ing object?
      if (value instanceof RegExp) {
        const result = { [REGEXP]: [value.source, value.flags] };
        const fullResult: EncodeResult = [result, transferables];
        seen.set(value, fullResult);
        return fullResult;
      }

      if (value instanceof URL) {
        const result = { [URL_ID]: value.href };
        const fullResult: EncodeResult = [result, transferables];
        seen.set(value, fullResult);
        return fullResult;
      }

      if (value instanceof Date) {
        const result = { [DATE]: value.toISOString() };
        const fullResult: EncodeResult = [result, transferables];
        seen.set(value, fullResult);
        return fullResult;
      }

      if (value instanceof Map) {
        const entries = [...value.entries()].map(([key, value]) => {
          return [encodeValue(key), encodeValue(value)];
        });
        const result = { [MAP]: entries };
        const fullResult: EncodeResult = [result, transferables];
        seen.set(value, fullResult);
        return fullResult;
      }

      if (value instanceof Set) {
        const entries = [...value].map((entry) => encodeValue(entry));
        const result = { [SET]: entries };
        const fullResult: EncodeResult = [result, transferables];
        seen.set(value, fullResult);
        return fullResult;
      }

      const valueIsIterator = isIterator(value);

      if (isBasicObject(value) || valueIsIterator) {
        const result: Record<string, any> = {};

        for (const key of Object.keys(value)) {
          result[key] = encodeValue((value as any)[key]);
        }

        if (valueIsIterator) {
          result.next ??= encodeValue((value as any).next.bind(value));
          result.return ??= encodeValue((value as any).return.bind(value));
          result.throw ??= encodeValue((value as any).throw.bind(value));
          result[ASYNC_ITERATOR] = true;
        }

        const fullResult: EncodeResult = [result, transferables];
        seen.set(value, fullResult);

        return fullResult;
      }
    }

    if (typeof value === "function") {
      const id = api.functions?.add(value);

      if (id == null) return [id];

      const result: EncodeResult = [{ [FUNCTION]: id }];
      seen.set(value, result);

      return result;
    }

    const result: EncodeResult = [value];
    seen.set(value, result);

    return result;
  }

  interface DecodeContext {
    api: ThreadEncoderApi;
    decode: Parameters<NonNullable<ThreadEncoderOptions["decode"]>>[1];
  }

  function decode(
    value: unknown,
    api: ThreadEncoderApi,
    retainedBy?: Iterable<MemoryRetainer>
  ) {
    const context: DecodeContext = {
      api,
      decode: (value: any) => decodeInternal(value, context, retainedBy, true),
    };

    return decodeInternal(value, context);
  }

  function decodeInternal(
    value: unknown,
    context: DecodeContext,
    retainedBy?: Iterable<MemoryRetainer>,
    isFromOverride = false
  ): any {
    const { api, decode } = context;

    if (!isFromOverride && decodeOverride) {
      return decodeOverride(value, decode, retainedBy);
    }

    if (typeof value === "object") {
      if (value == null) {
        return value as any;
      }

      if (Array.isArray(value)) {
        return value.map((value) => decodeInternal(value, context, retainedBy));
      }

      if (REGEXP in value) {
        return new RegExp(...(value as { [REGEXP]: [string, string] })[REGEXP]);
      }

      if (URL_ID in value) {
        return new URL((value as { [URL_ID]: string })[URL_ID]);
      }

      if (DATE in value) {
        return new Date((value as { [DATE]: string })[DATE]);
      }

      if (MAP in value) {
        return new Map(
          (value as { [MAP]: [any, any] })[MAP].map(([key, value]) => [
            decodeInternal(key, context, retainedBy),
            decodeInternal(value, context, retainedBy),
          ])
        );
      }

      if (SET in value) {
        return new Set(
          (value as { [SET]: any[] })[SET].map((entry) =>
            decodeInternal(entry, context, retainedBy)
          )
        );
      }

      if (FUNCTION in value) {
        const id = (value as { [FUNCTION]: string })[FUNCTION];

        const func = api.functions?.get(id);

        if (retainedBy && isMemoryManageable(func)) {
          for (const retainer of retainedBy) {
            retainer.add(func);
          }
        }

        return func;
      }

      if (!isBasicObject(value)) {
        return value;
      }

      const result: Record<string | symbol, any> = {};

      for (const key of Object.keys(value)) {
        if (key === ASYNC_ITERATOR) {
          result[Symbol.asyncIterator] = () => result;
        } else {
          result[key] = decodeInternal(
            (value as any)[key],
            context,
            retainedBy
          );
        }
      }

      return result;
    }

    return value;
  }
}

function isIterator(value: any) {
  return (
    value != null &&
    (Symbol.asyncIterator in value || Symbol.iterator in value) &&
    typeof (value as any).next === "function"
  );
}
