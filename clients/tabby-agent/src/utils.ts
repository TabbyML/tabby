export function sleep(milliseconds: number) {
  return new Promise((r) => setTimeout(r, milliseconds));
}

/**
 * @param obj Find a function in this object
 * @param keyPath A string of keys separated by dots, e.g 'foo.bar.getSomething'
 * @returns The function if found that has bound target context, null otherwise
 */
export function getFunction(obj, keyPath): Function | null {
  try {
    let [target, func] = keyPath.split(".").reduce(([_, obj], k) => [obj, obj[k]], [null, obj]);
    if (typeof func === "function") {
      return (func as Function).bind(target);
    }
  } catch (e) {
    // nothing
  }
  return null;
}
