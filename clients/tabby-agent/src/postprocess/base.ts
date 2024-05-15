import { CompletionItem } from "../CompletionSolution";
import { getLogger } from "../logger";

export type PostprocessFilterFactory = (config: unknown) => PostprocessFilter;
export type PostprocessFilter = (item: CompletionItem) => CompletionItem | Promise<CompletionItem>;
export const logger = getLogger("Postprocess");
