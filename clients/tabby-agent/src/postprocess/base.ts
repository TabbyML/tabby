import { CompletionResponse, CompletionContext } from "../CompletionContext";
import { rootLogger } from "../logger";

export type PostprocessFilter = (item: string, context: CompletionContext) => string | null | Promise<string | null>;

export const logger = rootLogger.child({ component: "Postprocess" });

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

export function applyFilter(
  filter: PostprocessFilter,
  context: CompletionContext,
): (response: CompletionResponse) => Promise<CompletionResponse> {
  return async (response: CompletionResponse) => {
    response.choices = (
      await Promise.all(
        response.choices.map(async (choice) => {
          const replaceLength = context.position - choice.replaceRange.start;
          const filtered = await filter(choice.text.slice(replaceLength), context);
          choice.text = choice.text.slice(0, replaceLength) + (filtered ?? "");
          return choice;
        }),
      )
    )
      .filter((choice) => !!choice.text)
      .distinct((choice) => choice.text);
    return response;
  };
}
