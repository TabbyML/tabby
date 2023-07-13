export function splitLines(input: string) {
  return input.match(/.*(?:$|\r?\n)/g).filter(Boolean); // Split lines and keep newline character
}

export function splitWords(input: string) {
  return input.match(/\w+|\W+/g).filter(Boolean); // Split consecutive words and non-words
}

export function isBlank(input: string) {
  return input.trim().length === 0;
}

// Using string levenshtein distance is not good, because variable name may create a large distance.
// Such as distance is 9 between `const fooFooFoo = 1;` and `const barBarBar = 1;`, but maybe 1 is enough.
// May be better to count distance based on words instead of characters.
import * as levenshtein from "fast-levenshtein";
export function calcDistance(a: string, b: string) {
  return levenshtein.get(a, b);
}

import { CancelablePromise } from "./generated";
export function cancelable<T>(promise: Promise<T>, cancel: () => void): CancelablePromise<T> {
  return new CancelablePromise((resolve, reject, onCancel) => {
    promise
      .then((resp: T) => {
        resolve(resp);
      })
      .catch((err: Error) => {
        reject(err);
      });
    onCancel(() => {
      cancel();
    });
  });
}
