declare const IS_BROWSER: boolean;
export const isBrowser = IS_BROWSER;

export function splitLines(input: string) {
  return input.match(/.*(?:$|\r?\n)/g).filter(Boolean) // Split lines and keep newline character
}

export function splitWords(input: string) {
  return input.match(/\w+|\W+/g).filter(Boolean); // Split consecutive words and non-words
}

export function isBlank(input: string) {
  return input.trim().length === 0;
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
