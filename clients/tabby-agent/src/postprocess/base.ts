import { CompletionRequest, CompletionResponse } from "../Agent";
import { splitLines } from "../utils";
import { rootLogger } from "../logger";

export type PostprocessContext = {
  request: CompletionRequest; // request contains full context, others are for easy access
  prefix: string;
  suffix: string;
  prefixLines: string[];
  suffixLines: string[];
};
export type PostprocessFilter = (item: string) => string | null | Promise<string | null>;

export const logger = rootLogger.child({ component: "Postprocess" });

export function buildContext(request: CompletionRequest): PostprocessContext {
  const prefix = request.text.slice(0, request.position);
  const suffix = request.text.slice(request.position);
  const prefixLines = splitLines(prefix);
  const suffixLines = splitLines(suffix);
  return {
    request,
    prefix,
    suffix,
    prefixLines,
    suffixLines,
  };
}

declare global {
  interface Array<T> {
    distinct(identity?: (x: T) => any): Array<T>;
  }
}

if (!Array.prototype.distinct) {
  Array.prototype.distinct = function <T>(this: T[], identity?: (x: T) => any): T[] {
    return [...new Map(this.map((item) => [identity?.(item) ?? item, item])).values()];
  };
}

export function applyFilter(filter: PostprocessFilter): (response: CompletionResponse) => Promise<CompletionResponse> {
  return async (response: CompletionResponse) => {
    response.choices = (
      await Promise.all(
        response.choices.map(async (choice) => {
          choice.text = await filter(choice.text);
          return choice;
        }),
      )
    )
      .filter((choice) => !!choice.text)
      .distinct((choice) => choice.text);
    return response;
  };
}
