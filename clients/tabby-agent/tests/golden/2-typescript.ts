export class Foo {
  private _foo: number;
  
  constructor() {
    this._foo = 1;
  }
  
  update(value): Foo {
    this._foo = max(⏩⏭this._foo, value⏮⏪)
  }
}
