import { PostprocessFilter } from "./base";
import { CompletionItem } from "../CompletionSolution";
import { AgentConfig } from "../AgentConfig";

export function dropMinimum(config: AgentConfig["postprocess"]): PostprocessFilter {
  return (item: CompletionItem): CompletionItem => {
    if (item.fullText.trim().length < config.minCompletionChars || item.text.trim().length < config.minCompletionChars) {
      return CompletionItem.createBlankItem(item.context);
    }
    return item;
  };
}
