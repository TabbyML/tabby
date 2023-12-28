import { CompletionResponse, CompletionResponseChoice, CompletionContext } from "../CompletionContext";
import { rootLogger } from "../logger";

type PostprocessFilterBase<T extends string | CompletionResponseChoice> = (
  input: T,
  context: CompletionContext,
) => T | null | Promise<T | null>;

export type PostprocessFilter = PostprocessFilterBase<string>;
export type PostprocessChoiceFilter = PostprocessFilterBase<CompletionResponseChoice>;

export const logger = rootLogger.child({ component: "Postprocess" });

declare global {
  interface Array<T> {
    distinct(identity?: (x: T) => any): Array<T>;
    mapAsync<U>(callbackfn: (value: T, index: number, array: T[]) => U | Promise<U>, thisArg?: any): Promise<U[]>;
  }
}

if (!Array.prototype.distinct) {
  Array.prototype.distinct = function <T>(this: T[], identity?: (x: T) => any): T[] {
    return [...new Map(this.map((item) => [identity?.(item) ?? item, item])).values()];
  };
}

if (!Array.prototype.mapAsync) {
  Array.prototype.mapAsync = async function <T, U>(
    this: T[],
    callbackfn: (value: T, index: number, array: T[]) => U | Promise<U>,
    thisArg?: any,
  ): Promise<U[]> {
    return await Promise.all(this.map((item, index) => callbackfn.call(thisArg, item, index, this)));
  };
}

export function applyFilter(
  filter: PostprocessFilter,
  context: CompletionContext,
): (response: CompletionResponse) => Promise<CompletionResponse> {
  return applyChoiceFilter(async (choice) => {
    const replaceLength = context.position - choice.replaceRange.start;
    const input = choice.text.slice(replaceLength);
    const filtered = await filter(input, context);
    choice.text = choice.text.slice(0, replaceLength) + (filtered ?? "");
    return choice;
  }, context);
}

export function applyChoiceFilter(
  choiceFilter: PostprocessChoiceFilter,
  context: CompletionContext,
): (response: CompletionResponse) => Promise<CompletionResponse> {
  return async (response: CompletionResponse) => {
    response.choices = (
      await response.choices.mapAsync(async (choice) => {
        return await choiceFilter(choice, context);
      })
    )
      .filter<CompletionResponseChoice>((choice): choice is NonNullable<CompletionResponseChoice> => {
        // Filter out empty choices.
        return !!choice && !!choice.text;
      })
      .distinct((choice) => choice.text);
    return response;
  };
}
