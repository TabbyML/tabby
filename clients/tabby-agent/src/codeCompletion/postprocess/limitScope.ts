import { PostprocessFilter } from "./base";
import { CompletionResultItem } from "../solution";
import { CompletionContext, CompletionExtraContexts } from "../contexts";
import { limitScopeByIndentation } from "./limitScopeByIndentation";

export function limitScope(): PostprocessFilter {
  return async (
    item: CompletionResultItem,
    context: CompletionContext,
    extraContext: CompletionExtraContexts,
  ): Promise<CompletionResultItem> => {
    return limitScopeByIndentation()(item, context, extraContext);
  };
}
