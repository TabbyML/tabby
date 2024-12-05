import type { CompletionItem } from "../solution";
import type { ConfigData } from "../../config/type";
import { getLogger } from "../../logger";

export type PostprocessFilterFactory =
  | (() => PostprocessFilter)
  | ((config: ConfigData["postprocess"]) => PostprocessFilter);
export type PostprocessFilter = (item: CompletionItem) => CompletionItem | Promise<CompletionItem>;
export const logger = getLogger("Postprocess");
