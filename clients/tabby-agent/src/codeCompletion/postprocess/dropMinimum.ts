import { PostprocessFilter } from "./base";
import { CompletionItem } from "../solution";
import { ConfigData } from "../../config/type";

export function dropMinimum(config: ConfigData["postprocess"]): PostprocessFilter {
  return (item: CompletionItem): CompletionItem => {
    if (
      item.fullText.trim().length < config.minCompletionChars ||
      item.text.trim().length < config.minCompletionChars
    ) {
      return CompletionItem.createBlankItem(item.context);
    }
    return item;
  };
}
