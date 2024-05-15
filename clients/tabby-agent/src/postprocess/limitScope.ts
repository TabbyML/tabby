import { CompletionItem } from "../CompletionSolution";
import { PostprocessFilter } from "./base";
import { limitScopeByIndentation } from "./limitScopeByIndentation";

export function limitScope(): PostprocessFilter {
  return async (item: CompletionItem): Promise<CompletionItem> => {
    return limitScopeByIndentation()(item);
  };
}
