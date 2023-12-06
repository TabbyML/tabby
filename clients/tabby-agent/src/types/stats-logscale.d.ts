// Ref: https://github.com/dallaylaen/stats-logscale-js/blob/main/lib/univariate.js
declare module "stats-logscale" {
  export class Univariate {
    constructor(args?: { base?: number; precision?: number; bins?: string });
    add(...data: number[]): this;
    count(): number;
    mean(): number | undefined;
    percentile(p: number): number | undefined;
  }
}
