export const noop = () => {
  //
};

export class Deferred<T> {
  public resolve: (value: T | PromiseLike<T>) => void = noop;
  public reject: (err?: unknown) => void = noop;
  public readonly promise: Promise<T>;

  constructor() {
    this.promise = new Promise<T>((resolve, reject) => {
      this.resolve = resolve;
      this.reject = reject;
    });
  }
}
