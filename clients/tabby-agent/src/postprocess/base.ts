import { CompletionItem } from "../CompletionSolution";
import { getLogger } from "../logger";
import { AgentConfig } from "../AgentConfig";

export type PostprocessFilterFactory = (() => PostprocessFilter) | ((config: AgentConfig["postprocess"]) => PostprocessFilter);
export type PostprocessFilter = (item: CompletionItem) => CompletionItem | Promise<CompletionItem>;
export const logger = getLogger("Postprocess");
