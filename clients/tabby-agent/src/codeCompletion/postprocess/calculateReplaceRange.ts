import { PostprocessFilter } from "./base";
import { CompletionItem } from "../solution";
import { calculateReplaceRangeByBracketStack } from "./calculateReplaceRangeByBracketStack";

export function calculateReplaceRange(): PostprocessFilter {
  return async (item: CompletionItem): Promise<CompletionItem> => {
    return calculateReplaceRangeByBracketStack(item);
  };
}
