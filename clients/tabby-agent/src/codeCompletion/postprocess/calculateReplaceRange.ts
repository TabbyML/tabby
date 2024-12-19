import { PostprocessFilter } from "./base";
import { CompletionItem } from "../solution";
import { calculateReplaceRangeByBracketStack } from "./calculateReplaceRangeByBracketStack";
import { calculateReplaceRangeBySemiColon } from "./calculateReplaceRangeBySemiColon";

export function calculateReplaceRange(): PostprocessFilter {
  return async (item: CompletionItem): Promise<CompletionItem> => {
    const afterBracket = calculateReplaceRangeByBracketStack(item);
    return calculateReplaceRangeBySemiColon(afterBracket);
  };
}
