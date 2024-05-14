import { PostprocessFilter } from "./base";
import { CompletionItem } from "../CompletionSolution";
import { isBlank } from "../utils";

export function dropBlank(): PostprocessFilter {
  return (item: CompletionItem): CompletionItem => {
    if (isBlank(item.fullText) || isBlank(item.text)) {
      return CompletionItem.createBlankItem(item.context);
    }
    return item;
  };
}
