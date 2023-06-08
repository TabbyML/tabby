import { CompletionRequest, CompletionResponse } from "./Agent";
import deepEqual from "deep-equal";
import { isBlank, splitLines } from "./utils";
import { rootLogger } from "./logger";

type PostprocessContext = CompletionRequest;
type PostprocessFilter = (item: string) => string | null | Promise<string | null>;

const logger = rootLogger.child({ component: "Postprocess" });

const removeDuplicateLines: (context: PostprocessContext) => PostprocessFilter = (context) => {
  return (input) => {
    const suffix = context.text.slice(context.position);
    const suffixLines = splitLines(suffix);
    const inputLines = splitLines(input);
    for (let index = Math.max(0, inputLines.length - suffixLines.length); index < inputLines.length; index++) {
      if (deepEqual(inputLines.slice(index), suffixLines.slice(0, input.length - index))) {
        logger.debug({ input, suffix, duplicateAt: index }, "Remove duplicate lines");
        return input.slice(0, index);
      }
    }
    return input;
  };
};

const dropBlank: PostprocessFilter = (input) => {
  return isBlank(input) ? null : input;
};

const applyFilter = (filter: PostprocessFilter) => {
  return async (response: CompletionResponse) => {
    response.choices = (
      await Promise.all(
        response.choices.map(async (choice) => {
          choice.text = await filter(choice.text);
          return choice;
        })
      )
    ).filter(Boolean);
    return response;
  };
};

export async function postprocess(
  request: CompletionRequest,
  response: CompletionResponse
): Promise<CompletionResponse> {
  return new Promise((resolve) => resolve(response))
    .then(applyFilter(removeDuplicateLines(request)))
    .then(applyFilter(dropBlank));
}
