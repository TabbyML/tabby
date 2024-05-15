import { PostprocessFilter } from "./base";
import { calculateReplaceRangeByBracketStack } from "./calculateReplaceRangeByBracketStack";
import { CompletionItem } from "../CompletionSolution";

export function calculateReplaceRange(): PostprocessFilter {
  return async (item: CompletionItem): Promise<CompletionItem> => {
    return calculateReplaceRangeByBracketStack(item);
  };
}
