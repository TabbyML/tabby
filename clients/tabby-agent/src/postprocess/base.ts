import { CompletionResponse, CompletionResponseChoice, CompletionContext } from "../CompletionContext";
import { getLogger } from "../logger";
import "../ArrayExt";

type PostprocessFilterBase<T extends string | CompletionResponseChoice> = (
  input: T,
  context: CompletionContext,
) => T | null | Promise<T | null>;

export type PostprocessFilter = PostprocessFilterBase<string>;
export type PostprocessChoiceFilter = PostprocessFilterBase<CompletionResponseChoice>;

export const logger = getLogger("Postprocess");

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
