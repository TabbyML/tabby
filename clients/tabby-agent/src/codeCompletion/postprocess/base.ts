import type { CompletionResultItem } from "../solution";
import type { CompletionContext, CompletionExtraContexts } from "../contexts";
import type { ConfigData } from "../../config/type";
import { getLogger } from "../../logger";

export type PostprocessFilterFactory =
  | (() => PostprocessFilter)
  | ((config: ConfigData["postprocess"]) => PostprocessFilter);

export type PostprocessFilter = (
  input: CompletionResultItem,
  context: CompletionContext,
  extraContext: CompletionExtraContexts,
) => CompletionResultItem | Promise<CompletionResultItem>;

export const logger = getLogger("Postprocess");
