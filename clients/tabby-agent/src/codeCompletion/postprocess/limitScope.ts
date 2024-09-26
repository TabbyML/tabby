import { PostprocessFilter } from "./base";
import { CompletionItem } from "../solution";
import { limitScopeByIndentation } from "./limitScopeByIndentation";

export function limitScope(): PostprocessFilter {
  return async (item: CompletionItem): Promise<CompletionItem> => {
    return limitScopeByIndentation()(item);
  };
}
