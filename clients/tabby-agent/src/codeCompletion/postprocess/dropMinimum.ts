import { PostprocessFilter } from "./base";
import { CompletionResultItem, emptyCompletionResultItem } from "../solution";
import { ConfigData } from "../../config/type";

export function dropMinimum(config: ConfigData["postprocess"]): PostprocessFilter {
  return (item: CompletionResultItem): CompletionResultItem => {
    if (item.text.trim().length < config.minCompletionChars) {
      return emptyCompletionResultItem;
    }
    return item;
  };
}
